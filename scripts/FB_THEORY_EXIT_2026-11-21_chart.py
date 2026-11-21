#!/usr/bin/env python3
"""FB_THEORY_EXIT_2026-11-21 — event-study chart + theory facts (research only).
Backtest fills (545, 5m) + real fills (1m, hourly marks): price vs entry over 72 h after the fill, split by today's outcome (TP / STOP).
Left: price path in % from entry (median + IQR band). Right: the same path as a fraction of the way from entry to the pre-pump base
(1.0 = at the base; < 0 = above entry).
Theory facts printed: base reached within 72 h, peak re-taken within 72 h, stopped fills that later traded back to −5 %.
Output: reports/FB_THEORY_EXIT_2026-11-21_chart.png + _data/facts.csv
Usage: venv/bin/python -I scripts/FB_THEORY_EXIT_2026-11-21_chart.py"""
import importlib.util, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_data")
HR = 3_600_000
import oos_real_flatbase_recon as RC  # noqa: E402


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


RL = _imp("FBR", "FB_THEORY_EXIT_2026-11-21_real.py")
SC = RL.SC


def facts_row(kind, pair, E, base, peak, t, h, l, c, i0, end_t, why, xt):
    last = int(np.searchsorted(t, end_t, "left"))
    hh, ll = h[i0:last], l[i0:last]
    tt = t[i0:last]
    after = tt >= xt
    return dict(kind=kind, pair=pair, why=why, base_hit=bool((ll <= base).any()), peak_hit=bool((hh >= peak).any()),
                back_to_tp_after_stop=bool(why == "STOP" and (ll[after] <= E * 0.95).any()),
                max_up=(hh.max() / E - 1) * 100, max_dn=(1 - ll.min() / E) * 100)


def main():
    A = pd.read_csv(os.path.join(D, "bt_anchors.csv"))
    B = pd.read_csv(os.path.join(D, "bt_cells.csv"))
    ref = B[(B.cell == "REF") & (B["mode"] == "row")].set_index("rid")
    hrs = np.arange(0, 73)
    P, Q, OUT, FACT = [], [], [], []
    for pair, g in A.groupby("pair"):
        x = RC.W.load(pair)
        if x is None:
            continue
        t, o, h, l, c = [np.asarray(a) for a in x[:5]]; t = t.astype("int64")
        for r in g.itertuples():
            i0 = int(np.searchsorted(t, int(r.cf_entry_t)))
            E = float(r.E)
            ix = np.searchsorted(t, int(r.cf_entry_t) + hrs * HR)
            ok = ix < len(t)
            p = np.full(len(hrs), np.nan); p[ok] = (c[np.minimum(ix[ok], len(t) - 1)] / E - 1) * 100
            P.append(p); Q.append((1 - (1 + p / 100)) / (1 - r.base / E)); OUT.append(ref.loc[r.rid, "why"])
            FACT.append(facts_row("BT", pair, E, r.base, r.peak, t, h, l, c, i0, int(r.cf_entry_t) + 72 * HR, ref.loc[r.rid, "why"],
                                  int(ref.loc[r.rid, "xt"])))
    P, Q, OUT = np.array(P), np.array(Q), np.array(OUT)
    # real fills
    RF = pd.read_csv(os.path.join(D, "real_fills.csv")); RCl = pd.read_csv(os.path.join(D, "real_cells.csv"))
    rref = RCl[RCl.cell == "REF"].set_index("fid")
    RP, RO = [], []
    for r in RF.itertuples():
        raw = RL.k1m(r.pair, r.te - HR, r.te + 72 * HR + 600_000)
        g = SC.grid(raw, (r.te - HR) // 60_000 * 60_000, int(raw["t"][-1]) + 60_000)
        i0 = int(np.searchsorted(g["t"], r.te // 60_000 * 60_000 + 60_000))
        base = r.E / (1 + r.dist_base / 100); peak = r.E * (1 + r.peak_d / 100)
        FACT.append(facts_row("REAL", r.pair, r.E, base, peak, g["t"], g["h"], g["l"], g["c"], i0, r.te + 72 * HR, rref.loc[r.fid, "why"],
                              int(rref.loc[r.fid, "xt"])))
        RP.append(np.array([np.nan if v is None else -v for v in json.loads(r.path)])[:73]); RO.append(rref.loc[r.fid, "why"])
    F = pd.DataFrame(FACT); F.to_csv(os.path.join(D, "facts.csv"), index=False)
    for k, g in F.groupby("kind"):
        print(k, "N", len(g), "· base reached ≤ 72 h", f"{g.base_hit.mean() * 100:.0f} %", "· pump peak re-taken ≤ 72 h", f"{g.peak_hit.mean() * 100:.0f} %",
              "· REF stops", int((g.why == "STOP").sum()), "of which back to −5 % later", int(g.back_to_tp_after_stop.sum()),
              "· stops that re-took the peak", int(((g.why == "STOP") & g.peak_hit).sum()))
    # chart
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#fcfcfb")
    col = {"TP": "#2a78d6", "STOP": "#eb6834"}
    for a in ax:
        a.set_facecolor("#fcfcfb"); a.grid(color="#e4e3df", lw=0.6); a.spines[["top", "right"]].set_visible(False)
    for w in ("TP", "STOP"):
        m = OUT == w
        md, lo, hi = np.nanmedian(P[m], 0), np.nanpercentile(P[m], 25, 0), np.nanpercentile(P[m], 75, 0)
        ax[0].fill_between(hrs, lo, hi, color=col[w], alpha=0.15, lw=0)
        ax[0].plot(hrs, md, color=col[w], lw=2, label=f"backtest, today's exit {w} (N {m.sum()}) — median + IQR")
        qm = np.nanmedian(np.clip(Q[m], -3, 3), 0)
        ax[1].plot(hrs, qm, color=col[w], lw=2, label=f"backtest {w} (N {m.sum()}) — median")
    for p_, w in zip(RP, RO):
        ax[0].plot(hrs[:len(p_)], p_, color=col.get(w, "#888"), lw=0.6, alpha=0.45)
    ax[0].plot([], [], color="#52514e", lw=0.6, label=f"real fills ({len(RP)}), thin, same colours")
    ax[0].axhline(-5, color="#1baf7a", ls="--", lw=1); ax[0].text(72, -5.4, "TP −5 %", ha="right", va="top", color="#0b0b0b")
    ax[0].axhline(10, color="#52514e", ls="--", lw=1); ax[0].text(72, 10.4, "stop +10 %", ha="right", color="#0b0b0b")
    med_base = -np.nanmedian(A.dist_base); med_peak = np.nanmedian((A.peak / A.E - 1) * 100); med_vw = np.nanmedian((A.vwap / A.E - 1) * 100)
    for v, lab in ((med_base, "median pre-pump base"), (med_peak, "median pump peak"), (med_vw, "median episode VWAP")):
        ax[0].axhline(v, color="#c3c2b7", lw=1); ax[0].text(1, v + 0.4, f"{lab} {v:+.1f} %", color="#52514e", fontsize=8.5)
    ax[0].set_ylim(-28, 26); ax[0].set_xlim(0, 72)
    ax[0].set_xlabel("hours after the fill"); ax[0].set_ylabel("price vs entry, %  (down = the short wins)")
    ax[0].set_title("FLATBASE shorts: price after the fill", loc="left", fontsize=11)
    ax[0].legend(loc="lower left", fontsize=8, frameon=False)
    ax[1].axhline(1, color="#c3c2b7", lw=1); ax[1].text(1, 1.03, "at the pre-pump base", color="#52514e", fontsize=8.5)
    ax[1].axhline(0, color="#c3c2b7", lw=1); ax[1].text(1, 0.03, "at entry", color="#52514e", fontsize=8.5)
    ax[1].set_xlim(0, 72); ax[1].set_ylim(-1.2, 1.2)
    ax[1].set_xlabel("hours after the fill"); ax[1].set_ylabel("share of the way from entry to the base")
    ax[1].set_title("How far the unwind goes toward the old base", loc="left", fontsize=11)
    ax[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_chart.png"), dpi=130, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
