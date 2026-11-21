#!/usr/bin/env python3
"""FB_EXIT_RERUN_2026-11-21 — reads (PREREG reports/FB_EXIT_RERUN_2026-11-21_PREREG.md). Research only.
Input: reports/FB_EXIT_RERUN_2026-11-21_data/{bt.csv, real.csv} (from FB_EXIT_RERUN_2026-11-21_sim.py)
Output: reports/FB_EXIT_RERUN_2026-11-21_tables.md + _chart.png
Usage: venv/bin/python -I scripts/FB_EXIT_RERUN_2026-11-21_eval.py"""
import importlib.util, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "FB_EXIT_RERUN_2026-11-21_data")
OUTMD = os.path.join(ROOT, "reports", "FB_EXIT_RERUN_2026-11-21_tables.md")
OUTPNG = os.path.join(ROOT, "reports", "FB_EXIT_RERUN_2026-11-21_chart.png")
sp = importlib.util.spec_from_file_location("FBT", os.path.join(ROOT, "scripts", "FB_THEORY_EXIT_2026-11-21_common.py"))
K = importlib.util.module_from_spec(sp); sp.loader.exec_module(K)
NOT, X, DAY, HR = 3050.0, 0.61, 86_400_000, 3_600_000
CAND = ["PKC12", "PKC15", "PKC20"]
SHOW = ["REF", "PKC12", "PKC15", "PKC20", "C_PK1", "K_S12", "K_S15", "K_S20"]
Z = 1.6449
L = []


def out(s=""):
    L.append(s); print(s)


def wilson_lo(k, n):
    if n == 0:
        return np.nan
    p = k / n
    return (p + Z * Z / (2 * n) - Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / (1 + Z * Z / n)


def kelly(p, aw, al):
    if al == 0 or np.isnan(al):
        return np.nan
    b = aw / abs(al)
    return p - (1 - p) / b


def summary(g, t0, t1):
    net = g.net.values; dur = (g.xt.values - g.te.values) / HR
    dcr, mdd, wd = K.dcr(net, g.xt.values, pd.Timestamp(t0, unit="ms"), pd.Timestamp(t1, unit="ms"), X)
    w, lo = net[net > 0], net[net <= 0]
    return dict(N=len(g), WR=(net > 0).mean() * 100, avg=net.mean(), usd=net.sum() * NOT / 100, avg_win=w.mean() if len(w) else np.nan,
                avg_loss=lo.mean() if len(lo) else np.nan, worst=net.min(), dcr=dcr, maxDD=mdd, worst_day=wd, dur_avg=dur.mean(),
                dur_med=np.median(dur), TP=int((g.why == "TP").sum()), STOP=int((g.why == "STOP").sum()),
                TIME=int(g.why.isin(["TIME", "OPEN"]).sum()))


def block_table(R, label, cells):
    t0, t1 = R.te.min(), R.xt.max()
    rows = []
    ref = R[R.cell == "REF"].set_index("key").net
    for c in cells:
        g = R[R.cell == c].sort_values("key")
        s = summary(g, t0, t1)
        dl = (g.set_index("key").net - ref.loc[g.key].values).values
        day = (g.te.values // DAY)
        m, lo, hi, P = K.block_boot(dl, day)
        s.update(dict(cell=c, d=m, ci=f"{lo:+.2f} … {hi:+.2f}" if np.isfinite(lo) else "", P=P,
                      saved=dl[dl > 0].sum() * NOT / 100, cut=-dl[dl < 0].sum() * NOT / 100, n_saved=int((dl > 1e-9).sum()),
                      n_cut=int((dl < -1e-9).sum()), loo_min=K.loo(dl)[0]))
        rows.append(s)
    T = pd.DataFrame(rows)
    cols = ["cell", "N", "WR", "avg", "usd", "avg_win", "avg_loss", "worst", "dcr", "maxDD", "worst_day", "dur_avg", "dur_med", "TP", "STOP",
            "TIME"]
    out(f"\n**{label} — before → after** (one size $3,050 / fill = 0.61 × equity; DCR one size 0.61 × equity, every calendar day; durations h)\n")
    out(K.md(T[cols]))
    out(f"\n**{label} — paired Δ vs today (cell − REF, %/fill), day-block bootstrap**\n")
    out(K.md(T[["cell", "d", "ci", "P", "n_saved", "saved", "n_cut", "cut", "loo_min"]]))
    return T


# ─────────────── load ───────────────
B = pd.read_csv(os.path.join(D, "bt.csv"))
B["key"] = B.rid
B["era"] = np.where(B.te < pd.Timestamp("2025-07-01").value // 10**6, "DESIGN", "HOLDOUT")
R = pd.read_csv(os.path.join(D, "real.csv"))
R["key"] = R.pair + "|" + R.k
Brow = B[B["mode"] == "row"]

# ─────────────── 1 · pick on DESIGN (frozen rule) ───────────────
des = Brow[Brow.era == "DESIGN"]
ref_d = des[des.cell == "REF"].set_index("rid").net
dm = {c: (des[des.cell == c].set_index("rid").net - ref_d).mean() for c in CAND}
best = max(dm.values())
pick = sorted([c for c in CAND if dm[c] >= best - 0.05], key=lambda c: float(c[3:]))[0]
out("# FB_EXIT_RERUN_2026-11-21 — tables (generated)\n")
out("**Step 1 — DESIGN pick (frozen rule: largest mean Δ, ties within 0.05 → tighter cap)**: "
    + " · ".join(f"{c} {dm[c]:+.3f}" for c in CAND) + f" → **PICK = {pick}**")
out(f"Controls on DESIGN: " + " · ".join(f"{c} {(des[des.cell == c].set_index('rid').net - ref_d).mean():+.3f}" for c in ("C_PK1", "K_S12", "K_S15", "K_S20")))

# ─────────────── 2 · backtest blocks ───────────────
out("\n## Backtest (545 fills, 5m, today's stack)")
TB = {}
for era in ("DESIGN", "HOLDOUT"):
    TB[era] = block_table(Brow[Brow.era == era].copy(), f"Backtest {era} ({len(Brow[(Brow.era == era) & (Brow.cell == 'REF')])})", SHOW)

# HOLDOUT gate set for every candidate (pick decides; others shown)
out("\n**HOLDOUT gate set** (pick is the decision row; others shown for context)\n")
H = Brow[Brow.era == "HOLDOUT"]; H20 = B[(B["mode"] == "s20") & (B.era == "HOLDOUT")]
grows = []
for c in CAND + ["C_PK1"]:
    a = H[H.cell == c].set_index("rid"); r = H[H.cell == "REF"].set_index("rid").loc[a.index]
    dl = a.net - r.net
    m, lo, hi, P = K.block_boot(dl.values, (a.te.values // DAY))
    hy = pd.to_datetime(a.te, unit="ms"); hy = hy.dt.year.astype(str) + "-H" + np.where(hy.dt.month <= 6, "1", "2")
    hm = dl.groupby(hy.values).mean()
    a20 = H20[H20.cell == c].set_index("rid"); r20 = H20[H20.cell == "REF"].set_index("rid").loc[a20.index]
    d20 = (a20.net - r20.net).mean()
    tot = dl.sum()
    pair_sh = dl.groupby(a.pair).sum().max() / tot if tot > 0 else np.nan
    day_sh = dl.groupby(a.te.values // DAY).sum().max() / tot if tot > 0 else np.nan
    sv, ct = dl[dl > 0].sum() * NOT / 100, -dl[dl < 0].sum() * NOT / 100
    t0, t1 = H.te.min(), H.xt.max()
    db = K.dcr(r.net.values, r.xt.values, pd.Timestamp(t0, unit="ms"), pd.Timestamp(t1, unit="ms"), X)[0]
    da = K.dcr(a.net.values, a.xt.values, pd.Timestamp(t0, unit="ms"), pd.Timestamp(t1, unit="ms"), X)[0]
    g = dict(ci=lo > 0, hy=(hm > 0).sum() >= 2, slip=d20 > 0, conc=(pair_sh < 0.5) and (day_sh < 0.5), sc=sv > ct, dcr=da >= db)
    grows.append(dict(cell=c, d=m, ci=f"{lo:+.2f} … {hi:+.2f}", P=P, halves=" / ".join(f"{k} {v:+.2f}" for k, v in hm.items()),
                      d_slip20=d20, top_pair=pair_sh, top_day=day_sh, saved=sv, cut=ct, dcr_b=db, dcr_a=da,
                      gates=f"{sum(g.values())}/6 " + "".join("✓" if v else "✗" for v in g.values())))
G = pd.DataFrame(grows)
out(K.md(G, 3))
out("gate order: CI>0 · ≥2/3 halves · 0.20 % slip · pair/day <50 % · saved>cut · DCR after≥before")

# ─────────────── 3 · real fills ───────────────
out("\n## Real fills (1m walk, real fill price, real funding)")
TR = {}
for blk in ("MASTER", "BATCH"):
    TR[blk] = block_table(R[R.block == blk].copy(), f"{blk} ({R[(R.block == blk) & (R.cell == 'REF')].shape[0]})", SHOW)

# per-fill tables
for blk in ("MASTER", "BATCH"):
    g = R[R.block == blk]
    ref = g[g.cell == "REF"].set_index("key"); pk = g[g.cell == pick].set_index("key"); c25 = g[g.cell == "C_PK1"].set_index("key")
    T = pd.DataFrame(dict(pair=ref.pair.str.replace("USDT", ""), opened=ref.k.str[:16], peak_above_entry=ref.peak_d, booked=ref.booked,
                          today=ref.net.round(2).astype(str) + " " + ref.why + f" · " + ((ref.xt - ref.te) / HR).round(1).astype(str) + " h",
                          stop_pick=pk.d, pick_res=pk.net.round(2).astype(str) + " " + pk.why + " · " + ((pk.xt - pk.te) / HR).round(1).astype(str) + " h",
                          delta=pk.net - ref.net, cap25=c25.net.round(2).astype(str) + " " + c25.why))
    T = T.sort_values("opened")
    out(f"\n**{blk} per fill — today (+10) vs {pick} (stop % used shown) vs C_PK1 (cap 25)**\n")
    out(K.md(T.reset_index(drop=True)))

# ─────────────── 4 · size / Kelly / clustering ───────────────
out("\n## Size read (FLATBASE 0.5 × 5× ≈ 0.61 × equity notional per fill)")
krow = []
for lab, src in (("MASTER real", R[R.block == "MASTER"]), ("Backtest ALL", Brow), ("Backtest HOLDOUT", Brow[Brow.era == "HOLDOUT"])):
    for c in ("REF", pick):
        g = src[src.cell == c]; net = g.net.values
        k, n = int((net > 0).sum()), len(net)
        aw = net[net > 0].mean(); al = net[net <= 0].mean() if (net <= 0).any() else np.nan
        # use backtest avg loss when the real block has no loss (needed for b)
        if np.isnan(al):
            bb = Brow[Brow.cell == c].net.values; al = bb[bb <= 0].mean(); al_src = "bt"
        else:
            al_src = "own"
        p, pf = k / n, wilson_lo(k, n)
        be = abs(al) / (aw + abs(al))
        kf, kfl = kelly(p, aw, al), kelly(pf, aw, al)
        stop_d = g.d.median() if c != "REF" else 10.0
        krow.append(dict(block=lab, cell=c, N=n, WR=p * 100, floor95=pf * 100, BE_WR=be * 100, avg_win=aw, avg_loss=al, loss_src=al_src,
                         kelly_obs=kf * 100, kelly_floor=kfl * 100, kelly_floor_notional=kfl / (abs(al) / 100) if kfl > 0 else 0.0,
                         stop_med=stop_d, worst=net.min(), cost_avg_stop_eq=abs(al) * X, cost_worst_eq=abs(net.min()) * X))
KT = pd.DataFrame(krow)
out("\nKelly = WR − (1 − WR) ÷ (avg win ÷ |avg loss|), as % of equity lost on a losing fill; kelly_floor_notional = Kelly-floor notional ÷ equity "
    "(today runs 0.61). cost_*_eq = % of equity lost at 0.61 × equity.\n")
out(K.md(KT))

out("\n**Clustering (backtest ALL 545, 0.61 × equity one size)**\n")
crow = []
for c in ("REF", pick, "C_PK1"):
    g = Brow[Brow.cell == c]
    st = g[g.why == "STOP"]
    perday = st.groupby(st.xt // DAY).size()
    per4h = st.groupby(st.te // (4 * HR)).size()
    dcr, mdd, wd = K.dcr(g.net.values, g.xt.values, pd.Timestamp(g.te.min(), unit="ms"), pd.Timestamp(g.xt.max(), unit="ms"), X)
    crow.append(dict(cell=c, stops=len(st), max_stops_same_day=int(perday.max()), days_with_2plus=int((perday >= 2).sum()),
                     max_stops_same_entry_window=int(per4h.max()), worst_day_eq=wd, maxDD_eq=mdd, dcr=dcr))
out(K.md(pd.DataFrame(crow)))
rm = R[(R.block == "MASTER") & (R.cell == "REF") & (R.why == "STOP")]
out(f"\nMASTER (real) today's-exit stops: {len(rm)} — closing days " + ", ".join(pd.to_datetime(rm.xt, unit='ms').dt.strftime('%m-%d %H:%M')))

open(OUTMD, "w").write("\n".join(L) + "\n")

# ─────────────── chart ───────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[17, 3.6]), facecolor="#fcfcfb")
for ax, blk in zip(axes, ("MASTER", "BATCH")):
    g = R[R.block == blk]
    ref = g[g.cell == "REF"].sort_values("k"); pk = g[g.cell == pick].set_index("key").loc[ref.key]
    x = np.arange(len(ref)); w = 0.38
    ax.set_facecolor("#fcfcfb")
    ax.bar(x - w / 2 - 0.01, ref.net, w, color="#a3a29c", label="today (stop +10 %)")
    ax.bar(x + w / 2 + 0.01, pk.net.values, w, color="#2a78d6", label=f"{pick}")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{p.replace('USDT', '')}\n{k[5:10]}" for p, k in zip(ref.pair, ref.k)], fontsize=7.5, color="#52514e")
    for xi, a, b in zip(x, ref.net.values, pk.net.values):
        if abs(a - b) > 1e-6:
            ax.annotate(f"{b - a:+.1f}", (xi, max(a, b) + 0.6 if max(a, b) > 0 else min(a, b) - 1.6), ha="center", fontsize=8, color="#0b0b0b")
    ax.set_title(f"{blk} ({len(ref)}) — net % per fill", fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-23, 9)
axes[0].set_ylabel("net % of notional (after fees, slip, funding)", color="#52514e")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle(f"FLATBASE_SHORT — today's stop (+10 %) vs {pick} (stop = pump peak × 1.01, capped) · labels = Δ pts", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUTPNG, dpi=130)
print("wrote", OUTMD, OUTPNG)
