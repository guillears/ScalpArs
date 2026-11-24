#!/usr/bin/env python3
"""📡 Route B — INTERIM Q3 look (new-sleeve triggers H1–H7), operator-ordered 2026-11-24 (PREREG Addendum 6). NOT A VERDICT.

Re-runnable at will:  venv/bin/python -I scripts/route_b_signals/interim_q3.py   (after build.py has refreshed the datasets)
Writes reports/ROUTE_B_INTERIM_Q3_<UTC date>.md (+ _keys.csv, _days.csv). Reads only the loader datasets (no network).

What it computes (PREREG §3 / §6 Q3 machinery, on every row recorded so far — the 12-10 DESIGN screen is NOT replaced by this):
  · cohort: H openers with a FINAL priced path, signal ≤ data end − 4 h (optional-stopping guard), recorded book (no_book out, §3);
  · per key: openers / day, separate hours (window units), the best of the 16 TP/SL cells (H3 / H5: the fixed TP 2 / SL 2 cell);
  · Δ vs the matched random minutes (same coin, same UTC day, same side, controls.py) = mean trigger net − mean control net, both on
    the coin-days holding both;
  · luck test: trigger / control tags shuffled within each coin-day (1,000×, seed 20261210), the best cell re-picked each time, the
    family maximum over the DESIGN-eligible keys (H1, H2, H4, H6, H7a/b/c + their "|c" variants) that reach the ≥ 20 / ≥ 5-day floor
    (PREREG Addendum 7 — counts only) × 16 cells → family p. Keys below the floor: best cell shown, not tested.
    H3 / H5 (HOLDOUT-only, fixed cell) get their own per-key shuffle p — shown, can never pass on this data;
  · hour-block bootstrap 95 % CI (2,000×) of the trigger mean and of Δ; the two time halves of the interim data; Kelly at the
    one-sided 95 % Wilson WR floor; top pair / top day share of the summed net; the same Δ without book_unwalkable rows (Addendum 1).
  · data completeness per UTC day (hours recorded, book coverage, final paths, control stubs).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "reports", "ROUTE_B_SIGNALS_data")
SEED = 20261210
N_PERM = 1000
N_BOOT = 2000
GUARD_H = 4
CELLS = [f"g_tp{a}_sl{b}_net" for a in (1, 2, 3, 5) for b in (1, 2, 3, 5)]
FIXED = "g_tp2_sl2_net"
HOLDOUT_ONLY = ("H3", "H5")
FLOOR_N = 20                 # A1-Q3 floor (§6): ≥ 20 possibilities on ≥ 5 days — Addendum 7 also uses it to define the luck family
FLOOR_DAYS = 5
HOUR_MS = 3_600_000
DAY_MS = 86_400_000


H7_START = "2026-11-21 22:02"      # DL 611 data (OI / full book): H7 and the "|c" depth variants start here


def fam(key):
    return str(key).split("|")[0].split("_")[0]


def wilson_lo(k, n, z=1.645):
    if n == 0:
        return math.nan
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def kelly_floor(x):
    x = np.asarray(x, float)
    w, l = x[x > 0], x[x < 0]                                   # zero-net rows count in N (Wilson), in neither average
    if not len(w) or not len(l):
        return math.nan
    b = w.mean() / abs(l.mean()) if l.mean() != 0 else math.inf
    p = wilson_lo(len(w), len(x))
    return p - (1 - p) / b


def load():
    T = pd.read_pickle(os.path.join(DATA, "triggers_H.pkl.gz"))
    C = pd.read_pickle(os.path.join(DATA, "controls_H.pkl.gz"))
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    return T, C, meta


def cohort(T, C, data_end, drop_unwalkable=False):
    cut = data_end - pd.Timedelta(hours=GUARD_H)
    t = T[(T.h_final == True) & (T.signal_time <= cut) & (T.h_cost_src != "no_book")].copy()  # noqa: E712
    if drop_unwalkable:
        t = t[t.h_cost_src != "book_unwalkable"]
    c = C[(C.ctrl_k >= 0) & (C.ctrl_final == True) & (C.h_final == True) & (C.h_cost_src != "no_book")  # noqa: E712
          & C.trigger_id.isin(t.trigger_id)].copy()
    if drop_unwalkable:
        c = c[c.h_cost_src != "book_unwalkable"]
    for d in (t, c):
        d["ms"] = d.signal_time.dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ms]").astype("int64")
        d["day"] = d.minute.astype("int64") // DAY_MS        # the TRIGGER minute's UTC day — the day the controls were drawn in
        d["hour"] = d.ms // HOUR_MS
        d["cd"] = d.pair.astype(str) + "|" + d.day.astype(str)
    return t, c


def matched(tk, ck):
    """rows of the coin-days holding both a trigger and a control (the Δ sample)."""
    both = set(tk.cd) & set(ck.cd)
    return tk[tk.cd.isin(both)], ck[ck.cd.isin(both)]


def delta_vec(tv, cv):
    return np.nanmean(tv, 0) - np.nanmean(cv, 0)


def perm_deltas(tk, ck, cells, rng, n=N_PERM):
    """null Δ per permutation × cell: tags shuffled within each coin-day (counts kept)."""
    tm, cm = matched(tk, ck)
    if not len(tm) or not len(cm):
        return np.full((n, len(cells)), np.nan)
    allv = pd.concat([tm[["cd"] + cells].assign(_t=1), cm[["cd"] + cells].assign(_t=0)], ignore_index=True)
    allv = allv.sort_values("cd", kind="stable").reset_index(drop=True)
    V = allv[cells].to_numpy(float)
    tags = allv._t.to_numpy()
    grp = allv.cd.to_numpy()
    starts = np.r_[0, np.flatnonzero(grp[1:] != grp[:-1]) + 1, len(grp)]
    out = np.empty((n, len(cells)))
    idx = np.arange(len(allv))
    for i in range(n):
        p = idx.copy()
        for a, b in zip(starts[:-1], starts[1:]):
            p[a:b] = rng.permutation(p[a:b])
        tg = tags[p]
        out[i] = np.nanmean(V[tg == 1], 0) - np.nanmean(V[tg == 0], 0)
    return out


def hour_boot(tk, ck, col, rng, n=N_BOOT):
    """95 % CI of the trigger mean and of Δ, resampling whole UTC hours (window units)."""
    tm, cm = matched(tk, ck)
    hrs = np.array(sorted(set(tk.hour)))
    if len(hrs) < 3:
        return (math.nan, math.nan), (math.nan, math.nan)
    d_ok = tm.hour.nunique() >= 3                                # a Δ CI needs ≥ 3 matched hours
    ts = tk.groupby("hour")[col].agg(["sum", "count"]).reindex(hrs, fill_value=0)
    mts = tm.groupby("hour")[col].agg(["sum", "count"]).reindex(hrs, fill_value=0)
    # controls are attached to their TRIGGER's hour (the unit is the trigger window)
    hour_of = dict(zip(tk.trigger_id, tk.hour))
    cm2 = cm.assign(th=cm.trigger_id.map(hour_of))
    mcs = cm2.groupby("th")[col].agg(["sum", "count"]).reindex(hrs, fill_value=0)
    A = ts.to_numpy(float); B = mts.to_numpy(float); Cc = mcs.to_numpy(float)
    m_s, d_s = [], []
    for _ in range(n):
        k = rng.integers(0, len(hrs), len(hrs))
        a = A[k].sum(0); b = B[k].sum(0); c = Cc[k].sum(0)
        m_s.append(a[0] / a[1] if a[1] else np.nan)
        d_s.append((b[0] / b[1] - c[0] / c[1]) if b[1] and c[1] else np.nan)
    q = lambda s: tuple(np.nanpercentile(s, [2.5, 97.5]))
    return q(m_s), (q(d_s) if d_ok else (math.nan, math.nan))


def key_row(key, tk, ck, col, rng, days_total):
    x = tk[col].to_numpy(float)
    tm, cm = matched(tk, ck)
    d = (tm[col].mean() - cm[col].mean()) if len(tm) and len(cm) else math.nan
    (m_lo, m_hi), (d_lo, d_hi) = hour_boot(tk, ck, col, rng)
    days = sorted(set(tk[tk.trigger_id.isin(set(ck.trigger_id))].day))     # days with an ELIGIBLE opener (≥ 1 final control)
    h1, h2 = halves_delta(tm, cm, col)
    s = tk[col].sum()
    pair_sh, day_sh = signed_share(tk, "pair", col), signed_share(tk, "day", col)
    return dict(key=key, family=fam(key), cell=col.replace("g_", "").replace("_net", ""), N=len(tk),
                per_day=len(tk) / days_total, days=len(days), hours=tk.hour.nunique(), coins=tk.pair.nunique(),
                WR=(x > 0).mean() if len(x) else math.nan, mean_net=x.mean() if len(x) else math.nan, ci_lo=m_lo, ci_hi=m_hi,
                N_matched=len(tm), N_ctrl=len(cm), ctrl_mean=cm[col].mean() if len(cm) else math.nan, delta=d,
                d_lo=d_lo, d_hi=d_hi, half1=h1, half2=h2, kelly_floor=kelly_floor(x), sum_net=s, top_pair_share=pair_sh,
                top_day_share=day_sh)


def halves_delta(tm, cm, col):
    """Δ in the two interim halves (split at the median of the matched trigger days; the frozen H-a / H-b split → 11-28 12:00 needs
    data past it — stated in the report). Controls follow their trigger's day."""
    days = sorted(set(tm.day))
    if len(days) < 2:
        return math.nan, math.nan
    mid = days[len(days) // 2]
    out = []
    for sel in (tm.day < mid, tm.day >= mid):
        ids = set(tm[sel].trigger_id)
        a, b = tm[sel][col], cm[cm.trigger_id.isin(ids)][col]
        out.append(a.mean() - b.mean() if len(a) and len(b) else math.nan)
    return tuple(out)


def signed_share(tk, by, col):
    """the largest group's share of the SIGNED net sum (B2's 'carries ≥ 50 % of the effect'); NaN when the sum is ≤ 0. Pure."""
    tot = tk[col].sum()
    if not len(tk) or not (tot > 0):
        return math.nan
    return float(tk.groupby(by)[col].sum().max() / tot)


def floor_keys(t, c, keys):
    """PREREG Addendum 7: the luck-test family = DESIGN-eligible keys whose ELIGIBLE openers (final, book-priced, ≥ 1 final priced
    control — Addendum 1 / 4) reach the A1-Q3 floor (≥ FLOOR_N on ≥ FLOOR_DAYS distinct UTC days). Counts only — no outcome. Pure."""
    out = []
    has_ctrl = set(c.trigger_id)
    for k in keys:
        tk = t[(t.trigger == k) & t.trigger_id.isin(has_ctrl)]
        if len(tk) >= FLOOR_N and tk.day.nunique() >= FLOOR_DAYS:
            out.append(k)
    return out


def family_null(t, c, fam_keys, rng, n=N_PERM):
    """observed Δ per cell for each family key + the family-max null (best cell re-picked per shuffle, max over keys)."""
    null_max = np.full(n, -np.inf)
    obs = {}
    for k in fam_keys:
        tk, ck = t[t.trigger == k], c[c.trigger == k]
        tm, cm = matched(tk, ck)
        if not len(tm) or not len(cm):
            continue
        obs[k] = delta_vec(tm[CELLS].to_numpy(float), cm[CELLS].to_numpy(float))
        null_max = np.maximum(null_max, np.nanmax(perm_deltas(tk, ck, CELLS, rng, n), 1))
    return obs, null_max


def run():
    T, C, meta = load()
    data_end = pd.Timestamp(meta["data_end"])
    t, c = cohort(T, C, data_end)
    tu, cu = cohort(T, C, data_end, drop_unwalkable=True)
    rng = np.random.default_rng(SEED)
    days_total = max(1e-9, (data_end - pd.Timedelta(hours=GUARD_H) - pd.Timestamp("2026-11-19 23:23", tz="UTC")).total_seconds() / 86400)
    days_total_h7 = max(1e-9, (data_end - pd.Timedelta(hours=GUARD_H) - pd.Timestamp(H7_START, tz="UTC")).total_seconds() / 86400)
    keys = sorted(T.trigger.unique(), key=lambda k: (fam(k), k))
    fam_keys = floor_keys(t, c, [k for k in keys if fam(k) not in HOLDOUT_ONLY])   # Addendum 7: counts only, before any outcome
    obs, null_max = family_null(t, c, fam_keys, rng)
    q95 = float(np.nanpercentile(null_max, 95)) if np.isfinite(null_max).any() else math.nan
    for k in keys:                                                              # best cell of keys below the floor: descriptive only
        if k not in obs and fam(k) not in HOLDOUT_ONLY:
            tm, cm = matched(t[t.trigger == k], c[c.trigger == k])
            if len(tm) and len(cm):
                obs[k] = delta_vec(tm[CELLS].to_numpy(float), cm[CELLS].to_numpy(float))

    rows = []
    for k in keys:
        tk, ck = t[t.trigger == k], c[c.trigger == k]
        if fam(k) in HOLDOUT_ONLY:
            col = FIXED
        elif k in obs and np.isfinite(obs[k]).any():
            col = CELLS[int(np.nanargmax(obs[k]))]
        else:
            col = FIXED
        r = key_row(k, tk, ck, col, rng, days_total_h7 if (fam(k).startswith("H7") or "|" in k) else days_total)
        r["N_eligible"] = int(tk.trigger_id.isin(set(ck.trigger_id)).sum())
        src = dict(zip(tk.trigger_id, tk.h_cost_src))
        r["N_ctrl_cost_mismatch"] = int((ck.trigger_id.map(src) != ck.h_cost_src).sum())
        r["N_raw"] = int((T.trigger == k).sum())
        r["N_nobook"] = int(((T.trigger == k) & (T.h_cost_src == "no_book")).sum())
        r["N_notfinal"] = int(((T.trigger == k) & (T.h_final != True)).sum())  # noqa: E712
        r["N_unwalkable"] = int((tk.h_cost_src == "book_unwalkable").sum())
        if fam(k) in HOLDOUT_ONLY:
            pdk = perm_deltas(tk, ck, [col], rng)[:, 0]
            r["luck_p"] = float(np.mean(pdk >= r["delta"])) if np.isfinite(r["delta"]) else math.nan
            r["luck_basis"] = "own shuffle (HOLDOUT-only key)"
        elif k in fam_keys:
            r["luck_p"] = float(np.mean(null_max >= r["delta"])) if np.isfinite(r["delta"]) else math.nan
            r["luck_basis"] = "family max"
        else:
            r["luck_p"] = math.nan
            r["luck_basis"] = "below floor (not in the luck test)"
        tku, cku = tu[tu.trigger == k], cu[cu.trigger == k]
        tm, cm = matched(tku, cku)
        r["delta_no_unwalk"] = (tm[col].mean() - cm[col].mean()) if len(tm) and len(cm) else math.nan
        r["floor_ok"] = k in fam_keys if fam(k) not in HOLDOUT_ONLY else bool(r["N_eligible"] >= FLOOR_N and r["days"] >= FLOOR_DAYS)
        rows.append(r)
    K = pd.DataFrame(rows).sort_values("per_day", ascending=False)

    # ── data completeness per UTC day ──
    T2 = T.copy()
    T2["day"] = T2.signal_time.dt.strftime("%Y-%m-%d")
    C2 = C.copy()
    C2["day"] = C2.signal_time.dt.strftime("%Y-%m-%d")
    D = T2.groupby("day").agg(openers=("trigger_id", "size"), no_book=("h_cost_src", lambda s: (s == "no_book").mean()),
                              unwalkable=("h_cost_src", lambda s: (s == "book_unwalkable").mean()),
                              final=("h_final", lambda s: (s == True).mean()),  # noqa: E712
                              hours_with_triggers=("signal_time", lambda s: s.dt.floor("h").nunique()))
    tday = dict(zip(T2.trigger_id, T2.day))
    st = C[C.ctrl_k < 0].trigger_id.map(tday).value_counts()
    D["ctrl_stubs"] = st.reindex(D.index).fillna(0).astype(int)
    cf = C2[C2.ctrl_k >= 0].groupby(C2.trigger_id.map(tday)).agg(ctrl_final=("ctrl_final", "mean"),
                                                                  missing_hours=("ctrl_day_missing_hours", "max"))
    D = D.join(cf)
    return K, D, meta, q95, data_end, days_total, len(t), len(c)


def fmt(x, p=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def write(K, D, meta, q95, data_end, days_total, nt, nc):
    date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    base = os.path.join(ROOT, "reports", f"ROUTE_B_INTERIM_Q3_{date}")
    K.to_csv(base + "_keys.csv", index=False)
    D.to_csv(base + "_days.csv")
    L = [f"# 📡 Route B — INTERIM Q3 look at the new-sleeve triggers ({date}) — **NOT A VERDICT**", "",
         f"Operator-ordered interim (PREREG Addendum 6). Data: recording start 2026-11-19 23:23 → {data_end:%Y-%m-%d %H:%M} UTC "
         f"(paths need 4 h, so signals to {data_end - pd.Timedelta(hours=GUARD_H):%m-%d %H:%M}) · {days_total:.1f} days · "
         f"{nt} final triggers with a recorded book · {nc} priced random-minute controls. Run: `scripts/route_b_signals/interim_q3.py`.",
         "", f"Luck bar (Addendum 7: family max of Δ over the H1/H2/H4/H6/H7 keys and variants that reach ≥ {FLOOR_N} trades on ≥ {FLOOR_DAYS} days, "
         f"eligible openers = with ≥ 1 final control, × 16 cells, 95th pct of 1,000 shuffles, keys shuffled independently): **{fmt(q95)} pp**"
         + ("" if q95 == q95 else " — **no key reaches the floor this run, so nothing is luck-tested**")
         + ". Keys below the floor show '–' (not tested). Halves = Δ in the two halves of the interim's matched days (the frozen H-a / H-b "
         "split → 11-28 12:00 needs data past it). Top pair · day share = the largest group's share of the SIGNED net sum ('–' when the "
         "sum ≤ 0); B2 fails at ≥ 50 %.", "",
         "| key | trades/day | N (with controls) · days with controls · hours | cell | WR | avg net % (hour CI) | random same coin | Δ vs random (hour CI) | luck p | halves | Kelly@floor | top pair · day share |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in K.itertuples():
        L.append(f"| {r.key} | {r.per_day:.1f} | {r.N} ({r.N_eligible}) · {r.days} · {r.hours} | {r.cell} | {r.WR*100:.0f} % | {fmt(r.mean_net)} "
                 f"({fmt(r.ci_lo)} … {fmt(r.ci_hi)}) | {fmt(r.ctrl_mean)} | **{fmt(r.delta)}** ({fmt(r.d_lo)} … {fmt(r.d_hi)}) | "
                 f"{fmt(r.luck_p, 3, False)} | {fmt(r.half1)} / {fmt(r.half2)} | {fmt(r.kelly_floor, 2)} | "
                 f"{fmt(r.top_pair_share*100, 0, False)} % · {fmt(r.top_day_share*100, 0, False)} % |")
    L += ["", "## Data completeness per UTC day", "",
          "| day | openers | no book | book unwalkable | path final | hours with triggers | control stubs | controls final | missing S3 hours |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in D.reset_index().itertuples():
        d = r.day
        L.append(f"| {d} | {int(r.openers)} | {r.no_book*100:.0f} % | {r.unwalkable*100:.0f} % | {r.final*100:.0f} % | {int(r.hours_with_triggers)} | "
                 f"{int(r.ctrl_stubs)} | {fmt(r.ctrl_final*100 if r.ctrl_final == r.ctrl_final else math.nan, 0, False)} % | "
                 f"{fmt(r.missing_hours, 0, False)} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    return base


if __name__ == "__main__":
    out = run()
    print(write(*out))
