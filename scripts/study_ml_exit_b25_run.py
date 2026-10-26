#!/usr/bin/env python3
"""ML EXIT DEEP study (2026-10-26, operator: "maybe the trades are good, the exit strategy is wrong — get more on winners, the same on
losers") — the per-fill engine: price path → ① post-exit run, ② MAE/MFE + timing, ③ every PRE-DECLARED exit variant through the
VALIDATED replica (scripts/ml_exit_optimize_yr5.py: 98.9 % of yr5 engine exits, 104/104 live master exits), + the AS-WAS replica
(fidelity on this cohort).

Evaluator = RP.evaluate copied with three ADDITIVE hooks (all default off → BASE reproduces RP.evaluate bit-exact; asserted at run):
  g_time  [(min since entry, k)]  runner giveback k×ATR switches to k from that minute on (time-based trail widening)
  g_peak  [(peak %, k)]           runner giveback switches to k once the running peak crossed the level (peak-based widening)
  td      (t0, t1, level)         TIME-DECAY stop for a fill not yet armed (peak < run_arm): the stop line rises linearly from the
                                  normal stop at t0 min to `level` at t1 min, flat after (label TIME_DECAY; not recovery-hold eligible)
Never-green exits use the replica's DOA hook (exit at the first tick ≥ N min when peak < x). Partials = 50 % BASE + 50 % the wider
variant (the % of a half-and-half position is the mean of the two halves' %; fees are % of notional, so they split the same way).

Usage:  venv/bin/python -W ignore scripts/study_ml_exit_b25_run.py [--jobs 10] [--yr5]
Outputs: reports/ML_EXIT_DEEP_2026-10-26_variants.csv (fill × variant), _fills.csv (per fill: path, post-exit, MAE/MFE),
         with --yr5: reports/ML_EXIT_DEEP_2026-10-26_yr5_variants.csv.gz (the backtest cross-check, NEW variants on the yr5 fills).
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_exit_b25_data as DD  # noqa: E402  (patches the replica's readers)
RP = DD.RP

OUT = os.path.join(DD.ROOT, "reports", "ML_EXIT_DEEP_2026-10-26")
HZ = 9 * 3600_000
POST_HZ = 4 * 3600_000


def V(**kw):
    p = dict(RP.BASE)
    p.update(dict(g_time=None, g_peak=None, td=None))
    p.update(kw)
    return p


LAD_LOOSE = [(2.0, .50), (3.0, .75), (4.0, 1.0)]
LAD_WIDE = [(1.25, .50), (1.5, .60), (2.0, .80), (3.0, 1.2), (4.0, 1.6)]
# ───────────── PRE-DECLARED GRID (written 2026-10-26 before any variant was run on this cohort) ─────────────
VARIANTS = {
    "BASE": V(),
    # W — winner side: keep winners longer
    "W_ARM0.6": V(run_arm=0.60), "W_ARM0.8": V(run_arm=0.80), "W_ARM1.0": V(run_arm=1.00),
    "W_K0.75": V(run_g_atr=0.75), "W_K1.25": V(run_g_atr=1.25), "W_K1.5": V(run_g_atr=1.5), "W_K2.0": V(run_g_atr=2.0),
    "W_LOCK0.0": V(run_lock=0.0), "W_LOCK0.2": V(run_lock=0.20), "W_LOCK0.3": V(run_lock=0.30),
    "W_LAD_off": V(ladder=[]), "W_LAD_loose": V(ladder=LAD_LOOSE), "W_LAD_wide": V(ladder=LAD_WIDE),
    "W_TW15_k1.5": V(g_time=[(15, 1.5)]), "W_TW30_k1.5": V(g_time=[(30, 1.5)]), "W_TW30_k2.0": V(g_time=[(30, 2.0)]),
    "W_PW1.0_k1.5": V(g_peak=[(1.0, 1.5)]), "W_PW1.0_k2.0": V(g_peak=[(1.0, 2.0)]),
    "W_HOLD360": V(noexp=360),
    # halves for the partials (not read alone — they exist to price "half at today's exit, half trailed wider")
    "_H_K2.0_noLAD": V(run_g_atr=2.0, ladder=[]),
    "_H_K3.0_noLAD_h360": V(run_g_atr=3.0, ladder=[], noexp=360),
    "_H_ARM1.0_K2.0_noLAD": V(run_arm=1.0, run_g_atr=2.0, ladder=[]),
    # L — loser side: cut losers earlier / tighter
    "L_SL-0.5": V(sl_fixed=-0.5), "L_SL-0.6": V(sl_fixed=-0.6), "L_SLatr1.0": V(sl_atr=1.0),
    "L_NG5_0.05": V(doa=(5, 0.05)), "L_NG5_0.10": V(doa=(5, 0.10)), "L_NG10_0.05": V(doa=(10, 0.05)),
    "L_NG10_0.10": V(doa=(10, 0.10)), "L_NG15_0.05": V(doa=(15, 0.05)), "L_NG15_0.10": V(doa=(15, 0.10)),
    "L_TD15_-0.4": V(td=(15, 15, -0.40)), "L_TD30_-0.2": V(td=(30, 30, -0.20)), "L_TDlin10-40_-0.2": V(td=(10, 40, -0.20)),
    # ── POST-HOC (added 2026-10-26 AFTER the grid above was read, motivated by §C: 23 % of stopped losers recover to +0.40 within
    #    60 min) — wider fixed stops, read ONLY on the live-stopped cohort (saved vs deeper; non-stopped fills must show Δ = 0).
    "X_SL-1.0": V(sl_fixed=-1.0), "X_SL-1.2": V(sl_fixed=-1.2),
}
POSTHOC = {"X_SL-1.0", "X_SL-1.2"}
PARTIALS = {"P_half_K2.0_noLAD": "_H_K2.0_noLAD", "P_half_K3.0_noLAD_h360": "_H_K3.0_noLAD_h360",
            "P_half_ARM1.0_K2.0_noLAD": "_H_ARM1.0_K2.0_noLAD"}
STATES = ["SBULL", "notSBULL", "ATR_lo", "ATR_mid", "ATR_hi"]      # state-conditional = each variant applied only in that state


def evaluate(P, fx, pr):
    """RP.evaluate + hooks g_time / g_peak / td (see module docstring). Kept line-for-line with RP.evaluate otherwise."""
    t, net = P["t"], P["net"]
    n = len(t)
    pk = np.maximum.accumulate(np.maximum(net, 0.0))
    pk_prev = np.r_[0.0, pk[:-1]]
    rel = t - fx["o_ms"]
    atr = fx["atr"] if (fx["atr"] and fx["atr"] > 0) else None
    if pr["sl_fixed"] is not None:
        S = np.full(n, float(pr["sl_fixed"]))
    else:
        S = np.where(P["sig"], pr["sl_sig"], pr["sl_base"]).astype(float)
        if atr is not None and pr["sl_atr"] > 0:
            S = np.minimum(S, -atr * pr["sl_atr"])
        if pr["sl_cap"] < 0:
            S = np.maximum(S, pr["sl_cap"])
        if pr["sl_quiet"] is not None and atr is not None and atr < pr["sl_quiet"][0]:
            S = np.minimum(S, pr["sl_quiet"][1])
    if pr["be_arm"] is not None:
        S = np.where(pk >= pr["be_arm"], np.maximum(S, pr["be_lock"]), S)
    S0 = S.copy()
    if pr.get("td") is not None:                                     # HOOK td
        t0, t1, lv = pr["td"]
        w = np.clip((rel - t0 * 60_000) / max((t1 - t0) * 60_000, 1), 0, 1)
        w = np.where(rel >= t0 * 60_000, np.where(t1 > t0, w, 1.0), 0.0)
        line = S + w * (lv - S)
        unarmed = pk < pr["run_arm"] - 0.005
        S = np.where(unarmed & (rel >= t0 * 60_000), np.maximum(S, line), S)
    stop_hit = net <= S + 0.01
    if pr["ladder"]:
        fl = np.full(n, -np.inf)
        for trg, off in pr["ladder"]:
            fl = np.where(pk_prev >= trg, np.maximum(fl, trg - off), fl)
        lad_hit = net <= fl
    else:
        lad_hit = np.zeros(n, bool)
    if pr["run_g_fixed"] is not None:
        g = np.full(n, float(pr["run_g_fixed"]))
    elif atr is not None:
        g = np.full(n, atr * pr["run_g_atr"])
        if pr.get("g_time"):                                         # HOOK g_time
            for mn, k in pr["g_time"]:
                g = np.where(rel >= mn * 60_000, atr * k, g)
        if pr.get("g_peak"):                                         # HOOK g_peak
            for lvl, k in pr["g_peak"]:
                g = np.where(pk >= lvl, atr * k, g)
    else:
        g = np.full(n, np.inf)
    if pr["run_g_frac"] is not None:
        g = np.minimum(g, pr["run_g_frac"] * pk)
    if pr.get("run_g_frac_only") is not None:
        g = pr["run_g_frac_only"] * pk
    rfl = pk - g
    if pr["run_lock"] is not None:
        rfl = np.maximum(rfl, pr["run_lock"])
    arm_lvl = pr["run_arm"]
    if pr.get("run_arm_atr") is not None and atr is not None:
        arm_lvl = pr["run_arm_atr"] * atr
    armed = pk >= arm_lvl - 0.005
    run_hit = armed & (net <= rfl)
    tp_hit = (net >= pr["tp"]) if pr["tp"] is not None else np.zeros(n, bool)
    e13_hit = (P["p"] < P["e13"]) & P["flip"] if pr["ema13"] else np.zeros(n, bool)
    tmax = min(pr["noexp"], pr["maxhold"]) * 60_000
    clk = rel >= tmax
    doa_hit = np.zeros(n, bool)
    if pr["doa"] is not None:
        dmin, dx = pr["doa"]
        j = RP._first(rel >= dmin * 60_000)
        if j is not None and pk[j] < dx:
            doa_hit[j] = True
    order = (("HARD_TP_LADDER", lad_hit), ("FIXED_TP", tp_hit), ("EMA13_CROSS", e13_hit), ("STOP", stop_hit),
             ("RUNNER_TRAIL", run_hit), ("DOA_EXIT", doa_hit), ("NO_EXPANSION", clk))

    def first_exit(start, rh_avail, prefix):
        best, why = None, None
        for name, m in order:
            k = RP._first(m, start)
            if k is not None and (best is None or k < best):
                best, why = k, name
        if best is None:
            return None
        if why == "STOP":
            why_full = "STOP_LOSS_WIDE" if P["sig"][best] and pr["sl_fixed"] is None else "STOP_LOSS"
            if pr["be_arm"] is not None and pk[best] >= pr["be_arm"] and S[best] >= pr["be_lock"] - 1e-9:
                why_full = "BE_LOCK"
            if S[best] > S0[best] + 1e-9:                             # HOOK td label
                return best, prefix + "TIME_DECAY"
            if rh_avail and pr["rh"] and why_full.startswith("STOP_LOSS"):
                held = _rh(best, S[best])
                if held is not None:
                    return held
            return best, prefix + why_full
        return best, prefix + why

    def _rh(i0, stop_level):
        hard = round(stop_level - 0.5, 4)
        if net[i0] <= hard + 0.01 or hard <= -2.2 or pk[i0] >= 0.40 or fx["rsi_entry"] is None:
            return None
        r0 = RP.btc_rsi_at(int(t[i0]))
        if r0 is None or not (r0 >= fx["rsi_entry"] and 60.0 <= r0 <= 66.0):
            return None
        t0_ = t[i0]
        hs = RP._first(net <= hard + 0.01, i0 + 1)
        rel_ = RP._first(pk >= 0.40, i0 + 1)
        tm = RP._first(((t - t0_) >= 30 * 60_000) & (net < 0), i0 + 1)
        tm2 = RP._first((t - t0_) >= 240 * 60_000, i0 + 1)
        cand = [(hs, "RH_HARD_STOP"), (tm, "RH_TIME_EXIT"), (tm2, "RH_TIME_EXIT"), (rel_, "RELEASE")]
        lim = min([k for k, _ in cand if k is not None] + [n - 1])
        bc = (int(t0_ // 300_000) + 1) * 300_000
        while bc <= t[lim]:
            r = RP.btc_rsi_at(bc + 2000)
            if r is None or r < 60.0 or r < fx["rsi_entry"]:
                k = RP._first(t >= bc + 2000, i0 + 1)
                if k is not None:
                    cand.append((k, "RH_PREMISE_EXIT"))
                break
            bc += 300_000
        cand = [(k, w) for k, w in cand if k is not None]
        if not cand:
            return None
        k, w = min(cand, key=lambda x: x[0])
        if w == "RELEASE":
            return first_exit(k, False, "RH_")
        return k, w

    r = first_exit(0, True, "")
    if r is None:
        return n - 1, "END_OF_DATA", float(net[-1])
    return r[0], r[1], float(net[r[0]])


# ───────────── per-fill descriptive metrics (post-exit run, MAE / MFE) ─────────────
def describe(P, fx, c_ms, X):
    t, p, net = P["t"], P["p"], P["net"]
    E = fx["E"]
    o = fx["o_ms"]
    out = dict(src=P["src"], path_end_min=(t[-1] - o) / 60_000)
    ix = int(np.searchsorted(t, c_ms, side="left"))
    life = slice(0, min(max(ix + 1, 1), len(t)))
    nl = net[life]; tl = t[life]
    j_pk, j_tr = int(np.argmax(nl)), int(np.argmin(nl))
    out.update(mfe=float(nl[j_pk]), mfe_min=(tl[j_pk] - o) / 60_000, mae=float(nl[j_tr]), mae_min=(tl[j_tr] - o) / 60_000)
    for N in (5, 10, 15):
        m = (t - o) <= N * 60_000
        out[f"peak_by{N}"] = float(net[m].max()) if m.any() else np.nan
        out[f"trough_by{N}"] = float(net[m].min()) if m.any() else np.nan
    # trough reached BEFORE the running peak first touched +0.40 (the "dip before winning")
    pk = np.maximum.accumulate(nl)
    ja = np.flatnonzero(pk >= 0.395)
    out["mae_before_arm"] = float(nl[: ja[0] + 1].min()) if len(ja) else float(nl.min())
    # post-exit: run measured from the LIVE exit price X in P&L points at 1× ((p − X) / E × 100)
    if ix >= len(t) or t[-1] < c_ms + 5 * 60_000:
        out["post_cov"] = False
        return out
    out["post_cov"] = True
    tp, pp = t[ix:], p[ix:]
    run = (pp - X) / E * 100
    for k in (5, 15, 30, 60):
        j = int(np.searchsorted(tp, c_ms + k * 60_000, side="right")) - 1
        out[f"post{k}"] = float(run[j]) if tp[-1] >= c_ms + k * 60_000 else np.nan
        m = tp <= c_ms + k * 60_000
        out[f"postmax{k}"] = float(run[m].max())
    lim = tp <= c_ms + POST_HZ
    rr = run[lim]
    rm = np.maximum.accumulate(np.maximum(rr, 0.0))
    for x in (0.3, 0.5, 1.0):
        hit = np.flatnonzero(rm - rr >= x)
        if len(hit):
            out[f"runpb{x}"] = float(rm[hit[0]]); out[f"runpb{x}_cens"] = False
            out[f"runpb{x}_min"] = (tp[hit[0]] - c_ms) / 60_000
        else:
            out[f"runpb{x}"] = float(rm[-1]); out[f"runpb{x}_cens"] = True
            out[f"runpb{x}_min"] = np.nan
    return out


def run_fill(fx, variants, c_ms=None, X=None):
    res, desc = {}, None
    pending = dict(variants)
    for hz in (2 * 3600_000, 6 * 3600_000, HZ):
        P = RP.prep_path(fx, hz)
        if P is None:
            break
        if hz == HZ and c_ms is not None:
            desc = describe(P, fx, c_ms, X)
        for name, pr in list(pending.items()):
            k, why, pct = evaluate(P, fx, pr)
            if why != "END_OF_DATA" or hz == HZ:
                res[name] = (why, pct, int(P["t"][k]), P["src"])
                pending.pop(name)
        if not pending and (c_ms is None or desc is not None):
            break
    if desc is None and c_ms is not None:
        P = RP.prep_path(fx, HZ)
        desc = describe(P, fx, c_ms, X) if P is not None else dict(src="none")
    for name in pending:
        res[name] = ("NO_DATA", np.nan, None, "none")
    return res, desc


def _work(batch):
    out_v, out_d = [], []
    for fx, c_ms, X, aswas in batch:
        vv = dict(VARIANTS)
        if aswas is not None:
            vv["ASWAS"] = dict(aswas, g_time=None, g_peak=None, td=None)
        try:
            res, desc = run_fill(fx, vv, c_ms, X)
            # bit-exact guard: BASE through the copied evaluator == RP.evaluate
            P = RP.prep_path(fx, 2 * 3600_000)
            if P is not None:
                k0, w0, p0 = RP.evaluate(P, fx, RP.BASE)
                k1, w1, p1 = evaluate(P, fx, VARIANTS["BASE"])
                if w0 != "END_OF_DATA" and (w0 != w1 or abs(p0 - p1) > 1e-12):
                    res["_GUARD_FAIL"] = (f"{w0}|{w1}", p1 - p0, None, "")
        except Exception as ex:
            res = {v: (f"ERR:{ex}", np.nan, None, "err") for v in vv}
            desc = dict(src="err")
        for v, (why, pct, tex, src) in res.items():
            out_v.append(dict(key=fx["key"], variant=v, reason=why, pct=pct, exit_ms=tex, src=src))
        if desc is not None:
            out_d.append(dict(key=fx["key"], **desc))
    return out_v, out_d


def _pool(items, jobs):
    items = sorted(items, key=lambda x: (x[0]["pair"], x[0]["o_ms"]))
    n = max(1, len(items) // (jobs * 6))
    batches = [items[i:i + n] for i in range(0, len(items), n)]
    V_, D_ = [], []
    with ProcessPoolExecutor(jobs) as ex:
        for a, b in ex.map(_work, batches):
            V_.extend(a); D_.extend(b)
    return pd.DataFrame(V_), pd.DataFrame(D_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--yr5", action="store_true")
    a = ap.parse_args()
    if a.yr5:
        Y = RP.load_yr()
        Y = Y[~Y.cb]
        items = [(dict(f, key=f["key"]), None, None, None) for f in RP.fills_of(Y)]
        Vv, _ = _pool(items, a.jobs)
        Vv.to_csv(OUT + "_yr5_variants.csv.gz", index=False)   # gzip: 6 MB → ~0.5 MB in git
        print(f"yr5: {Y.shape[0]} fills × {Vv.variant.nunique()} variants → {OUT}_yr5_variants.csv.gz")
        return
    C = DD.cohort()
    items = []
    for r in C.itertuples():
        fx = DD.fill_dict(r)
        items.append((fx, int(r.c_ms), float(r.X), RP.aswas_params(int(r.o_ms))))
    Vv, Dd = _pool(items, a.jobs)
    Vv.to_csv(OUT + "_variants.csv", index=False)
    Dd.to_csv(OUT + "_fills_paths.csv", index=False)
    g = Vv[Vv.variant == "_GUARD_FAIL"]
    print(f"{len(C)} fills × {Vv.variant.nunique()} variants; evaluator guard failures: {len(g)}; errors: "
          f"{Vv.reason.astype(str).str.startswith('ERR').sum()}")


if __name__ == "__main__":
    main()
