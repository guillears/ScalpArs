#!/usr/bin/env python3
"""MAX_HOLD_STUDY_2026-11-21 — BACKTEST walker: today's exit with a MAX-HOLD grid, real funding charged (short sleeves).

Research only (no bot code / config / git touched). Operator 11-21: "Should we review the max duration per sleeve, considering funding?
(24 h in EXH)". Grid FROZEN here, before any result was read:
  EXH / DIST : cap h ∈ {6, 8, 12, 16, 20, 24 (today), 30, 36, 48}
  FB         : cap h ∈ {24, 36, 48, 60, 72 (today), 96, 120}
Re-uses the DL 616 machinery verbatim (scripts/FUNDING_STUDY_ALL_2026-11-21_bt.py: same cohorts, same 1m / 5m sources, same funding
pricer C.settle_on_bars, same costs) — only the path is loaded further (EXH / DIST to entry + 49 h on 1m, FB to entry + 121 h on 5m) and
the time exit is evaluated at every cap:
  * TP / stop are cap-independent events on the path (EXH / DIST price stop; FB stop counts funding PAID, DL 607 — the line at bar j only
    depends on the funding settled by bar j). A fill whose TP / stop comes before the cap exits there under EVERY longer cap → Δ = 0.
  * A fill still open at the cap closes at MARKET at the close of the last bar before entry + cap: taker 0.05 % + the row's slip (≥ 0.10 %
    = the DL 603 / 609 / 616 convention); sensitivity at 0.02 % (typical live) and 0.30 % (thin coin) is stored for the time exits only.
  * Funding = Σ real settlements in (fill, exit bar] (mark-scaled), so a longer hold pays (or receives) every extra settlement.
  * Parity: the 24 h (EXH / DIST) and 72 h (FB) cells must reproduce bt_<SLEEVE>.pkl base_net of DL 616 on the same rid.
Output: reports/MAX_HOLD_STUDY_2026-11-21_data/bt_<SLEEVE>.pkl
Usage: venv/bin/python -I scripts/MAX_HOLD_STUDY_2026-11-21_bt.py EXH|DIST|FB [--procs 6]"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "MAX_HOLD_STUDY_2026-11-21_data")
HR, MIN, DAY = 3_600_000, 60_000, 86_400_000
CAPS = {"EXH": (6, 8, 12, 16, 20, 24, 30, 36, 48), "DIST": (6, 8, 12, 16, 20, 24, 30, 36, 48), "FB": (24, 36, 48, 60, 72, 96, 120)}
TODAY = {"EXH": 24, "DIST": 24, "FB": 72}


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


FSB = _imp("FSB", "FUNDING_STUDY_ALL_2026-11-21_bt.py")      # DL 616 walker / cohorts (imported, never edited)
C = FSB.C
G = FSB.G


def cap_walk(t, o, h, l, c, i0, E, tp, sl, T0, caps, F, slip, entry_cost, taker, maker, paid_stop, nb, through=0.05):
    """one short from bar i0 → {cap: (net, why, hold_h, fund, net_s02, net_s30, censored)} + path label / reach.
    Same event logic as FUNDING_STUDY_ALL bt.walk_all (stop first on the HIGH at max(line, open) + slip; maker TP ≥ `through` % through)."""
    tt, oo, hh, ll, cc, Fj = t[i0:], o[i0:], h[i0:], l[i0:], c[i0:], F[i0:]
    n = len(tt)
    if n < 2:
        return None
    line = E * (1 + (sl + (np.minimum(Fj, 0.0) if paid_stop else np.zeros(n))) / 100.0)
    Tp = E * (1 - tp / 100.0)
    hs = hh >= line; tl = ll <= Tp * (1 - through / 100.0)
    ks = int(np.argmax(hs)) if hs.any() else n; kt = int(np.argmax(tl)) if tl.any() else n
    if ks < n and ks <= kt:
        kev, wev = ks, "STOP"; Xev = max(line[ks], oo[ks]) * (1 + slip / 100); fev = taker * Xev / E
    elif kt < n:
        kev, wev = kt, "TP"; Xev = Tp; fev = maker * Xev / E
    else:
        kev, wev, Xev, fev = n, None, None, None
    T00 = int(tt[0]); bar = nb * MIN
    out = {}
    for H in caps:
        end_t = T0 + H * HR
        last = int(np.searchsorted(tt, end_t, "left")) - 1
        cens = int(tt[-1]) + bar < end_t              # path ends before the cap: the TIME cell is an OPEN mark
        if kev <= last:
            net = (1 - Xev / E) * 100 - entry_cost - fev + Fj[kev]
            out[H] = (net, wev, (int(tt[kev]) - T00) / HR + nb / 60, float(Fj[kev]), net, net, False)
        else:
            last = max(last, 0)
            res = []
            for s in (slip, 0.02, 0.30):
                X = cc[last] * (1 + s / 100)
                res.append((1 - X / E) * 100 - entry_cost - taker * X / E + Fj[last])
            out[H] = (res[0], "TIME", (int(tt[last]) - T00) / HR + nb / 60, float(Fj[last]), res[1], res[2], cens)
    fav = (1 - ll / E) * 100; adv = (hh / E - 1) * 100
    lim = int(np.searchsorted(tt, T0 + TODAY_H[0] * HR, "left"))
    up = np.flatnonzero(fav[:lim] >= 1.0); dn = np.flatnonzero(adv[:lim] >= sl)
    iu = up[0] if len(up) else lim; idn = dn[0] if len(dn) else lim
    lab = "WORKED" if iu < idn else ("NEVER" if idn < iu else "FLAT")
    return out, lab


TODAY_H = [24]


def flat(rid, pair, T, res, lab, extra):
    out, lab_ = res, lab
    row = dict(rid=rid, pair=pair, te=T, label=lab_, **extra)
    for H, (net, why, hold, fund, n02, n30, cens) in out.items():
        row.update({f"c{H}_net": net, f"c{H}_why": why, f"c{H}_hold": hold, f"c{H}_fund": fund, f"c{H}_n02": n02, f"c{H}_n30": n30,
                    f"c{H}_cens": cens})
    return row


# ───────────────────────────── EXH / DIST (1m) ─────────────────────────────
def ed_pair(args):
    p, g, sleeve = args
    load1m, TB = G["load1m"], G["TB"]
    caps = CAPS[sleeve]; Hmax = max(caps)
    days = sorted({int(T) // DAY + kk for T in g.entry_ms.values for kk in range(4)})
    one, okd = load1m(p, days)
    five = None
    ft, fc, months = TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fc = np.asarray(fc, float)
    rates = np.diff(fc) if len(fc) > 1 else np.zeros(0)
    out = []
    for r in g.itertuples():
        T = int(r.entry_ms); need = {(T + k * HR) // DAY for k in range(0, Hmax + 1, 6)}; nb = 1
        use1 = one is not None and need <= okd
        if use1:
            t1 = one[0]; a = int(np.searchsorted(t1, T)); use1 = a < len(t1) and t1[a] == T
        if use1:
            b = int(np.searchsorted(t1, T + (Hmax + 1) * HR)); t, o, h, l, c = [x[a:b] for x in one]
        else:
            if five is None:
                five = TB.load(p)
            if five is None:
                continue
            t5 = np.asarray(five[0], "int64"); a = int(np.searchsorted(t5, T))
            if a >= len(t5) or t5[a] != T:
                continue
            b = int(np.searchsorted(t5, T + (Hmax + 1) * HR)); t, o, h, l, c = t5[a:b], *[np.asarray(five[k][a:b], float) for k in (1, 2, 3, 4)]
            nb = 5
        if len(o) < 2:
            continue
        E = float(o[0]); sl = min(1.5 * r.atrp, 6.0); tp = min(20 * r.atrp, G["cap"])
        mE = pd.Timestamp(T, unit="ms").strftime("%Y-%m"); m24 = pd.Timestamp(T + 24 * HR, unit="ms").strftime("%Y-%m")
        mX = pd.Timestamp(T + Hmax * HR, unit="ms").strftime("%Y-%m")
        miss = (mE not in months) or (m24 not in months) or not len(ft)          # = DL 616's exclusion
        miss_ext = miss or (mX not in months)                                    # the longest cap also needs its month
        Fb = np.zeros(len(t)) if miss else C.settle_on_bars(t, o, E, T, ft, rates, nb, markscale=True)
        w = cap_walk(t, o, h, l, c, 0, E, tp, sl, T, caps, Fb, r.slip, r.slip + 0.05, 0.05, 0.02, False, nb)
        if w is None:
            continue
        out.append(flat(r.rid, p, T, *w, dict(miss=miss, miss_ext=miss_ext, nb=nb, tp=tp, sl=sl, slip=r.slip)))
    return out


# ───────────────────────────── FB (5m rebuild) ─────────────────────────────
def fb_pair(args):
    pair, g, sleeve = args
    OC, RC, F = G["OC"], G["RC"], G["F"]
    caps = CAPS["FB"]; Hmax = max(caps)
    x = RC.W.load(pair)
    out = []
    if x is None:
        return out
    t, o, h, l, c = [np.asarray(a) for a in x[:5]]; t = t.astype("int64")
    tidx = {int(tt): k for k, tt in enumerate(t)}
    fx = F.get(pair)
    for r in g.itertuples():
        i = tidx.get(int(r.cf_entry_t))
        if i is None:
            continue
        E = float(r.cf_entry_px); te = int(r.cf_entry_t) + 10_000
        _, miss = OC.fund(F, pair, te, int(t[i]) + 72 * HR)
        _, miss_x = OC.fund(F, pair, te, int(t[i]) + Hmax * HR)
        if miss or fx is None or not len(fx[0]):
            Fb = np.zeros(len(t))
        else:
            Fb = C.settle_on_bars(t, o, E, te, fx[0], fx[3], 5, markscale=True)
        j1 = int(np.searchsorted(t, int(t[i]) + (Hmax + 1) * HR))
        w = cap_walk(t[:j1], o[:j1], h[:j1], l[:j1], c[:j1], i, E, 5.0, 10.0, int(t[i]), caps, Fb[:j1], float(r.slip_x), RC.TAKER, RC.TAKER,
                     RC.MAKER, True, 5)
        if w is None:
            continue
        out.append(flat(r.rid, pair, int(r.cf_entry_t), *w, dict(miss=miss, miss_ext=miss or miss_x, nb=5, tp=5.0, sl=10.0,
                                                                     slip=float(r.slip_x))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sleeve"); ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    TODAY_H[0] = TODAY[A.sleeve]
    if A.sleeve == "FB":
        FSB.fb_setup(); X = FSB.fb_rows(); fn = fb_pair
    else:
        FSB.ed_setup(A.sleeve); X = FSB.ed_rows(A.sleeve); fn = ed_pair
    jobs = [(p, g, A.sleeve) for p, g in X.groupby("pair")]
    res = []
    with mp.get_context("fork").Pool(A.procs) as pool:
        for k, o in enumerate(pool.imap_unordered(fn, jobs, chunksize=2)):
            res += o
            if k % 100 == 0:
                print(k, "/", len(jobs), flush=True)
    R = pd.DataFrame(res)
    keep = [c for c in ("rid", "univ", "xi") if c in X.columns]
    R = R.merge(X[keep], on="rid", how="left")
    # parity vs DL 616 (same rid, today's cap)
    ref = pd.read_pickle(os.path.join(ROOT, "reports", "FUNDING_STUDY_ALL_2026-11-21_data", f"bt_{A.sleeve}.pkl"))[["rid", "base_net", "base_why"]]
    P = R.merge(ref, on="rid")
    H0 = TODAY[A.sleeve]
    gap = (P[f"c{H0}_net"] - P.base_net).abs()
    print(A.sleeve, "rows", len(R), "of", len(X), "· miss", int(R.miss.sum()), "· miss_ext", int(R.miss_ext.sum()),
          "· 5m", int((R.nb == 5).sum()), f"· parity vs DL 616 on {len(P)}: max|Δ| {gap.max():.6f}, why match {(P[f'c{H0}_why'] == P.base_why).mean():.4f}")
    R.to_pickle(os.path.join(OUT, f"bt_{A.sleeve}.pkl"))


if __name__ == "__main__":
    main()
