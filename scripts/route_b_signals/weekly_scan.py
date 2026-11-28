#!/usr/bin/env python3
"""📡 Route B — the WEEKLY RESEARCH SCAN runner (reports/ROUTE_B_WEEKLY_SCAN_PREREG.md §6–§7 + Addenda 1–10). Research only: no
engine, config, recorder or AWS change; nothing arms (gates only alert). Reads the loader datasets (run build.py first); network only
for the H4 frozen lines (h4_frozen_lines.py) and the event-study chart of a PASSED line.

Run (the scheduled task, Thursdays from 2026-12-10, after `build.py --no-report`):
    venv/bin/python -I scripts/route_b_signals/weekly_scan.py            # the phase is decided by the date + the files on disk
    venv/bin/python -I scripts/route_b_signals/weekly_scan.py --dry-run  # COUNTS ONLY (cohorts, coverage, test families) — any day

Phases (§7, Addendum 10):
  · before 2026-12-10 00:00 UTC: nothing is read (exit 0). --dry-run prints counts only — never an outcome.
  · first run on / after 12-10, no SURVIVORS.json yet: the DESIGN screen, ONCE (Q1 exhaustive screen + rotation null, Q2 lines
    (A2), Q3 trigger keys (A1-Q3), Q4 cuts frozen on DESIGN market days) → reports/ROUTE_B_WEEKLY_SCAN_SURVIVORS.json + the report.
    It stops there: A3 says the survivor list is COMMITTED before any HOLDOUT row is read, and a scheduled run never commits.
  · every later run: refuses to read HOLDOUT unless SURVIVORS.json is in git HEAD and unmodified; then the HOLDOUT status of every
    listed line (B1 floors → COLLECTING; the first read with the floors met is FROZEN in reports/ROUTE_B_WEEKLY_SCAN_state.json and
    reported unchanged forever), the exit re-check of not-yet-judged Q1 / Q2 survivors (STALE), Q4 / Q5 descriptives, the H4 frozen
    lines (Addendum 9) → reports/ROUTE_B_WEEKLY_SCAN_<UTC date>.md.
Pure helpers are unit-tested in tests/test_route_b_weekly_scan.py on synthetic data.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402
import interim_q3 as I  # noqa: E402
import q2 as Q2  # noqa: E402

ROOT, DATA = CM.ROOT, CM.DATA
PREREG = os.path.join(ROOT, "reports", "ROUTE_B_WEEKLY_SCAN_PREREG.md")
SURVIVORS = os.path.join(ROOT, "reports", "ROUTE_B_WEEKLY_SCAN_SURVIVORS.json")
STATE = os.path.join(ROOT, "reports", "ROUTE_B_WEEKLY_SCAN_state.json")
H4_STATE = os.path.join(ROOT, "reports", "H4_FROZEN_LINES_state.json")

SEED = 20261210
N_ROT = 1000
N_BOOT = 2000
FIRST_RUN = pd.Timestamp("2026-12-10 00:00", tz="UTC")
DESIGN_START = pd.Timestamp("2026-11-19 23:23", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-12-07 00:00", tz="UTC")
HALF_SPLIT = pd.Timestamp("2026-11-28 12:00", tz="UTC")
HALF_SPLIT_611 = pd.Timestamp("2026-11-29 11:00", tz="UTC")       # DL 611 columns, H7 and the "|c" variants (§2)
DL611_START = pd.Timestamp(I.H7_START, tz="UTC")
DL630_START = pd.Timestamp("2026-11-23 14:33:21", tz="UTC")       # S5's sample (Addendum 1 item 3)

SLEEVES = ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FLATBASE_SHORT", "FRENZY_WILLY_HT")     # §7 order (no-spot = S5, after)
SHORT = {"EXHAUSTION_SHORT": "EXH", "DISTRIBUTION_SHORT": "DIST", "FLATBASE_SHORT": "FLATBASE", "FRENZY_WILLY_HT": "WILLY_HT"}
HOLD_H = {"EXHAUSTION_SHORT": 24, "DISTRIBUTION_SHORT": 24, "FRENZY_WILLY_HT": 24, "FLATBASE_SHORT": 72}   # §2 guard
TRIG_HOLD_H = 4
KEEP_CLASSES = ("taken", "filter", "capacity")
CORE_GATES = ("EXH_NO_SPOT", "EXH_VOL24_LOW", "EXH_VOL_DECAY", "DIST_NOT_VSW", "DIST_VOL24_LOW", "NOT_FLAT")   # §2, named
A_MIN_N, A_MIN_DAYS = 10, 5
B_MIN_N, B_MIN_DAYS = 15, 8
Q3_B_N, Q3_B_DAYS = 30, 10
Q4_B_DAYS = 10
A2_P = 0.10
HOLDOUT_ONLY_VARS = ("rb_imb_05", "fb_imb_05", "rb_taker_buy_share_5", "rb_taker_buy_share_15", "taker_bs_ratio",
                     "taker_bs_ratio_chg60_pct")                   # §0 / Addendum 1 item 1
PRE_SEEN_GATE_INPUTS = ("j_exh_vol24_usd", "j_exh_has_spot", "j_exh_vol_decay", "j_exh_ssc", "j_exh_oi_vol", "j_dist_vsw_mult",
                        "j_dist_bar_ret_pct", "j_dist_vol24_usd", "j_dist_pos_chg_4h", "j_dist_rsi1h")   # §0 rows 4–5: their gate
#                         outcomes were printed (taken vs blocked per gate) → HOLDOUT-only like the §0 imbalance families (Addendum 10)
DL611_PREFIX = ("fb_", "oi_", "top_ls_", "global_ls_", "taker_bs_")
VAR_PREFIX = ("rb_",) + DL611_PREFIX + ("j_", "journal_")
Q2_DESIGN = ("S1a", "S2b", "S2c", "S3a", "S3b", "S4a", "S5")
Q2_ALL = ("S1a", "S1b", "S1c", "S2a", "S2b", "S2c", "S3a", "S3b", "S4a", "S4b", "S5")
Q4_NATURAL_ZERO = ("funding_mean_pct", "premium_mean_pct")
COLOUR = {"PASSED": "🟢", "SURVIVED DESIGN": "🟡", "STALE": "🟡", "FAILED": "🔴", "CONTRADICTED": "🔴", "COLLECTING": "⚪",
          "UNTESTABLE": "⚪"}


# ───────────────────────────────────────────── cohorts (pure) ─────────────────────────────────────────────
def gate_of(status):
    s = str(status)
    return s.split(":", 1)[1] if ":" in s else s


def _bool(s):
    """True only for a real True / 1 / "True"; NaN / NA / None / anything else → False. Pure."""
    def one(v):
        if v is None or v is pd.NA or (isinstance(v, float) and v != v):
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return v == 1
        return str(v) == "True"
    return s.map(one).astype(bool)


def _prep(d, y_col, net_col="out_net_pct"):
    d = d.copy()
    d["y"] = pd.to_numeric(d[y_col], errors="coerce") if y_col in d else np.nan
    d["net"] = pd.to_numeric(d[net_col], errors="coerce") if net_col in d else np.nan
    d["day"] = d.signal_time.dt.strftime("%Y-%m-%d")
    d["win"] = d.window15.astype(str) if "window15" in d else d.signal_time.dt.floor("15min").astype(str)
    return d.sort_values(["signal_time"] + (["signal_id"] if "signal_id" in d else []), kind="stable").reset_index(drop=True)


def sleeve_cohort(S, sleeve):
    """§2 Q1 / Q2 cohort: possibilities (poss_dup False) meeting the core rule, then taken / filter-blocked / capacity-refused; every
    gate §2 names as the sleeve's core rule is out (the loader's block_class files EXH_NO_SPOT / VOL24 / VOL_DECAY and DIST_VOL24_LOW
    as 'filter' — §2 governs). y = the WORKED label (out_lbl_worked), net = out_net_pct. Pure."""
    g = S.status.map(gate_of)
    m = ((S.sleeve == sleeve) & ~_bool(S.poss_dup) & _bool(S.core_rule_met) & S.block_class.isin(KEEP_CLASSES)
         & ~g.isin(CORE_GATES) & (g != "SKIPPED_CONFIRM"))
    return _prep(S[m], "out_lbl_worked")


def s5_cohort(S):
    """S5's DL 630 cohort: EXH rows blocked by EXH_NO_SPOT from the DL 630 commit, one per possibility; y = s5_lbl_worked. Pure."""
    m = ((S.sleeve == "EXHAUSTION_SHORT") & (S.status.astype(str) == "BLOCKED:EXH_NO_SPOT") & ~_bool(S.poss_dup)
         & (S.signal_time >= DL630_START))
    return _prep(S[m], "s5_lbl_worked")


def guard(d, cutoff, hold_h):
    """the optional-stopping guard: signal ≤ the scan cutoff − the max hold. Pure."""
    return d[d.signal_time <= cutoff - pd.Timedelta(hours=hold_h)]


def design(d):
    return d[(d.signal_time >= DESIGN_START) & (d.signal_time < HOLDOUT_START)]


def holdout(d):
    return d[d.signal_time >= HOLDOUT_START]


def labelled(d):
    return d[d.y.notna()].reset_index(drop=True)


# ───────────────────────────────────────────── Q1 test family (pure) ─────────────────────────────────────────────
def num(d, c):
    if c not in d:
        return np.full(len(d), np.nan)
    s = d[c]
    if s.dtype == object:
        s = s.map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v)
    return pd.to_numeric(s, errors="coerce").to_numpy(float)


def excluded_var(c):
    """Addendum 10's exclusions, as written: price levels and data-quality stamps (rb_mid, rb_mark, journal_t, *_has_* / has_*,
    *_age_min, rb_tape_minutes_*, *_lb). Pure."""
    return (c in ("rb_mid", "rb_mark", "journal_t", "j_bl_sole", "j_exh_universe") or "_has_" in c or c.startswith("has_") or c.endswith("_age_min")
            or c.startswith("rb_tape_minutes_") or c.endswith("_lb"))


def screen_vars(d):
    """every coin-level rb_* / DL 611 column + every entry stamp (j_* / journal_*) the rows carry, numeric, ≥ 2 distinct scored values;
    → (vars, skipped {var: why}). Pure."""
    out, skip = [], {}
    for c in d.columns:
        if not c.startswith(VAR_PREFIX) or excluded_var(c):
            continue
        x = num(d, c)
        sc = ~np.isnan(x)
        if not sc.any():
            skip[c] = "never scored / not numeric"
        elif len(np.unique(x[sc])) < 2:
            skip[c] = "constant"
        else:
            out.append(c)
    return out, skip


def is_dl611(defn):
    return any(str(defn.get(k, "")).startswith(DL611_PREFIX) for k in ("var", "var2"))


def is_holdout_only(defn):
    return any(defn.get(k) in HOLDOUT_ONLY_VARS + PRE_SEEN_GATE_INPUTS for k in ("var", "var2"))


def sides(X, defn):
    """(A, B) boolean arrays of a frozen test definition on rows whose variables are X {var: float array}. A = positive / top / the
    named quadrant; B = non-positive / bottom / the other three quadrants; unscored rows in neither. Pure."""
    x = X[defn["var"]]
    sc = ~np.isnan(x)
    if defn["kind"] == "1d":
        if defn["gran"] == "sign":
            return sc & (x > 0), sc & (x <= 0)
        return sc & (x >= defn["hi"]), sc & (x <= defn["lo"])
    y = X[defn["var2"]]
    sc &= ~np.isnan(y)
    a = sc & ((x > defn["med"]) == bool(defn["hx"])) & ((y > defn["med2"]) == bool(defn["hy"]))
    return a, sc & ~a


def make_tests(D, vars_):
    """every 1D test (sign · tercile · quintile, cuts = DESIGN quantiles, pandas linear) and every 2D quadrant (each variable's
    DESIGN median) of `vars_` on the labelled DESIGN rows D → (defs, A, B) with A / B bool matrices (n × K). Pure (no outcome)."""
    X = {v: num(D, v) for v in vars_}
    defs, As, Bs = [], [], []

    def add(defn):
        a, b = sides(X, defn)
        defs.append(defn)
        As.append(a)
        Bs.append(b)
    med = {}
    for v in vars_:
        x = X[v]
        sc = ~np.isnan(x)
        if not sc.any():
            continue
        if (x[sc] > 0).any() and (x[sc] <= 0).any():
            add(dict(kind="1d", var=v, gran="sign"))
        for gran, (ql, qh) in (("tercile", (1 / 3, 2 / 3)), ("quintile", (0.2, 0.8))):
            lo, hi = (float(q) for q in pd.Series(x[sc]).quantile([ql, qh]))
            if lo < hi:
                add(dict(kind="1d", var=v, gran=gran, lo=lo, hi=hi))
        med[v] = float(np.median(x[sc]))
    vs = [v for v in vars_ if v in med]
    for i, v in enumerate(vs):
        for w in vs[i + 1:]:
            for hx in (1, 0):
                for hy in (1, 0):
                    add(dict(kind="2d", var=v, var2=w, med=med[v], med2=med[w], hx=hx, hy=hy))
    n = len(D)
    A = np.array(As, bool).T if As else np.zeros((n, 0), bool)
    B = np.array(Bs, bool).T if Bs else np.zeros((n, 0), bool)
    return defs, A, B


def onehot(codes):
    u, inv = np.unique(np.asarray(codes).astype(str), return_inverse=True)
    M = np.zeros((len(u), len(inv)))
    M[inv, np.arange(len(inv))] = 1.0
    return M


def floor_mask(A, B, days, n_min=A_MIN_N, d_min=A_MIN_DAYS):
    """A1 / B1 count floor per test: ≥ n_min rows and ≥ d_min distinct UTC days on EACH side. Counts only. Pure."""
    if not A.shape[1]:
        return np.zeros(0, bool)
    Dm = onehot(days)
    da, db = ((Dm @ A) > 0).sum(0), ((Dm @ B) > 0).sum(0)
    return (A.sum(0) >= n_min) & (B.sum(0) >= n_min) & (da >= d_min) & (db >= d_min)


def effects(y, A, B):
    """WORKED-rate difference mean(y | A) − mean(y | B) per test (NaN where a side is empty). Pure."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    nA, nB = A.sum(0), B.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (y @ A) / nA - (y @ B) / nB


def cluster_t(y, A, B, Gm):
    """effect and its window-clustered t per test: influence-function (CR1) variance of the two-sample mean difference, clusters =
    the rows' window15 (Gm = one-hot, clusters × rows). SE 0 → t NaN. Pure."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    nA, nB = A.sum(0), B.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mA, mB = (y @ A) / nA, (y @ B) / nB
        U = A * (y[:, None] - mA) / nA - B * (y[:, None] - mB) / nB
        Sg = Gm @ U
        G = ((Gm @ (A + B)) > 0).sum(0)
        var = (Sg ** 2).sum(0) * G / np.maximum(G - 1, 1)
        se = np.sqrt(var)
        t = np.where(se > 1e-12, (mA - mB) / se, np.nan)
    return mA - mB, t


def rotation_offset_min(days):
    """the rotation's minimum offset in rows = the largest number of rows any one UTC day holds ("at least one UTC day of rows"). Pure."""
    return int(pd.Series(np.asarray(days).astype(str)).value_counts().max()) if len(days) else 0


def rotation_null(y, A, B, Gm, days, rng, n=N_ROT):
    """family-maximum |t| under n rotations of the outcome series (rows in time order) by a random offset in [d, len − d], wrapping.
    None when the cohort is too short to rotate (len < 2d + 1). Pure given rng."""
    d, m = rotation_offset_min(days), len(y)
    if not A.shape[1] or m < 2 * d + 1 or d == 0:
        return None
    out = np.empty(n)
    A, B = np.asarray(A, float), np.asarray(B, float)
    for i in range(n):
        k = int(rng.integers(d, m - d + 1))
        _, t = cluster_t(np.roll(y, k), A, B, Gm)
        out[i] = np.nanmax(np.abs(t)) if np.isfinite(t).any() else 0.0
    return out


def eff1(y, a, b):
    """one test's effect (NaN-safe on y). Pure."""
    ok = ~np.isnan(y)
    a, b = a & ok, b & ok
    return float(y[a].mean() - y[b].mean()) if a.any() and b.any() else math.nan


def half_masks(d, dl611):
    split = HALF_SPLIT_611 if dl611 else HALF_SPLIT
    st = d.signal_time
    return (st < split).to_numpy(), (st >= split).to_numpy()


def _sign(x):
    return 0 if not (x == x) or x == 0 else (1 if x > 0 else -1)


def a1_checks(D, a, b, eff, dl611):
    """A1-Q1's non-null items for one test: halves same sign, NET same way (floors are checked by the caller). Pure."""
    y, net = D.y.to_numpy(float), D.net.to_numpy(float)
    h1, h2 = half_masks(D, dl611)
    e1, e2 = eff1(y, a & h1, b & h1), eff1(y, a & h2, b & h2)
    en = eff1(net, a, b)
    return dict(half_a=e1, half_b=e2, net_eff=en, halves_ok=_sign(e1) == _sign(eff) == _sign(e2) != 0,
                net_ok=_sign(en) == _sign(eff) != 0, net_priced=en == en)


def net_verdict(net_eff, need_sign):
    """§3 NET agreement: 'ok' · 'unpriced' (a side has no NET — a data gap, never a contradiction) · 'contra'. Pure."""
    if not net_eff == net_eff:
        return "unpriced"
    return "ok" if _sign(net_eff) == need_sign != 0 else "contra"


def design_q1(d_all, sleeve, rng, n_rot=None):
    """the Q1 DESIGN screen for one sleeve (A1-Q1) → dict(summary, survivors, bar). d_all = the sleeve cohort, guard applied."""
    D = labelled(design(d_all))
    vars_, skip = screen_vars(design(d_all))
    cov = {v: float(np.mean(~np.isnan(num(design(d_all), v)))) for v in vars_}
    out = dict(sleeve=sleeve, n_cohort=int(len(design(d_all))), n_labelled=int(len(D)), n_vars=len(vars_), skipped=skip,
               low_coverage=sorted(v for v, c in cov.items() if c < 0.5), coverage=cov, survivors=[], dropped=[], bar=None, n_tests=0,
               n_family=0, design_ids=D.signal_id.tolist() if "signal_id" in D else [])
    if len(D) < 2 * A_MIN_N:
        out["note"] = "too few labelled DESIGN rows for any test to reach the floor"
        return out
    defs, A, B = make_tests(D, vars_)
    fl = floor_mask(A, B, D.day.to_numpy())
    out["n_tests"], out["n_family"] = len(defs), int(fl.sum())
    if not fl.any():
        out["note"] = "no test reaches the ≥ 10 / ≥ 5-day floor"
        return out
    A, B = A[:, fl], B[:, fl]
    defs = [x for x, k in zip(defs, fl) if k]
    y = D.y.to_numpy(float)
    Gm = onehot(D.win.to_numpy())
    eff, t = cluster_t(y, A, B, Gm)
    null = rotation_null(y, A, B, Gm, D.day.to_numpy(), rng, n_rot or N_ROT)
    if null is None:
        out["note"] = "cohort too short to rotate (< 2 days of rows + 1)"
        return out
    bar = float(np.percentile(null, 95))
    out["bar"], out["best_abs_t"] = bar, float(np.nanmax(np.abs(t)))
    for j in np.flatnonzero(np.abs(np.nan_to_num(t)) > bar):
        a, b = A[:, j], B[:, j]
        ck = a1_checks(D, a, b, eff[j], is_dl611(defs[j]))
        row = dict(defs[j], line=q1_line_id(sleeve, defs[j]), eff=float(eff[j]), t=float(t[j]), nA=int(a.sum()), nB=int(b.sum()),
                   holdout_only=is_holdout_only(defs[j]), **ck)
        nv = net_verdict(ck["net_eff"], _sign(eff[j]))
        if ck["halves_ok"] and nv == "ok":
            out["survivors"].append(dict(row, status="SURVIVED DESIGN"))
        elif not ck["halves_ok"]:                                  # §6 D: every status is reported
            out["dropped"].append(dict(row, status="FAILED", note="beat the bar; DESIGN halves disagree"))
        elif nv == "unpriced":
            out["dropped"].append(dict(row, status="UNTESTABLE", note="beat the bar; NET unpriced on a side"))
        else:
            out["dropped"].append(dict(row, status="CONTRADICTED", note="beat the bar; NET points the other way (§3)"))
    return out


def q1_line_id(sleeve, d):
    if d["kind"] == "1d":
        return f"Q1|{SHORT[sleeve]}|{d['var']}|{d['gran']}"
    return f"Q1|{SHORT[sleeve]}|{d['var']}×{d['var2']}|{'hi' if d['hx'] else 'lo'},{'hi' if d['hy'] else 'lo'}"


# ───────────────────────────────────────────── bootstrap / B-gate helpers (pure) ─────────────────────────────────────────────
def boot_eff(y, a, b, clusters, rng, n=N_BOOT):
    """bootstrap draws of mean(y | a) − mean(y | b), resampling whole clusters (window15 rows, or days). Pure given rng."""
    ok = ~np.isnan(y)
    a, b = a & ok, b & ok
    codes, _ = pd.factorize(pd.Series(np.asarray(clusters).astype(str)))
    G = codes.max() + 1 if len(codes) else 0
    if G < 2:
        return np.full(n, np.nan)
    yz = np.where(ok, y, 0.0)
    an, ay = np.bincount(codes, a, G), np.bincount(codes, a * yz, G)
    bn, by = np.bincount(codes, b, G), np.bincount(codes, b * yz, G)
    K = rng.integers(0, G, (n, G))
    with np.errstate(invalid="ignore", divide="ignore"):
        return ay[K].sum(1) / an[K].sum(1) - by[K].sum(1) / bn[K].sum(1)


def ci95(draws):
    x = np.asarray(draws, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) >= 10 else (math.nan, math.nan)


def loo_min_ratio(y, a, b, groups, eff):
    """min over groups of effect(without the group) ÷ effect — B2's leave-one-out (keeps the sign and ≥ 50 % ⇔ ratio ≥ 0.5) and its
    concentration item (a group carries ≥ 50 % of the effect ⇔ ratio ≤ 0.5). A group whose removal empties a side carries the whole
    effect (ratio 0). NaN when eff is 0 / NaN. Pure."""
    if not (eff == eff) or eff == 0:
        return math.nan
    g = np.asarray(groups).astype(str)
    r = []
    for u in np.unique(g[a | b]):
        k = g != u
        e = eff1(y, a & k, b & k)
        r.append(e / eff if e == e else 0.0)
    return float(min(r)) if r else math.nan


def line_rng(line):
    """one generator per line (seed 20261210 + crc32 of the line id): a frozen verdict reproduces on its own, whatever other lines
    drew before it. Pure."""
    return np.random.default_rng(SEED + zlib.crc32(str(line).encode()) % (2 ** 31))


def regime_signs(d):
    s = pd.to_numeric(d.get("btc_d_trend", pd.Series(dtype=float)), errors="coerce").dropna()
    return sorted({int(np.sign(v)) for v in s if v != 0})


def b_gate(H, a, b, design_eff, need_sign, rng, *, ratio_to_design=True, cluster_col="win"):
    """B1 + B2 (+ B4 label) for one Q1 / Q2 line on its HOLDOUT rows H (labelled). need_sign = the sign the line must show (DESIGN's,
    or the predicted one for a HOLDOUT-only line). ratio_to_design False → only the sign is required (HOLDOUT-only). Pure given rng."""
    y, net = H.y.to_numpy(float), H.net.to_numpy(float)
    days = H.day.to_numpy()
    nA, nB = int(a.sum()), int(b.sum())
    dA, dB = len(set(days[a])), len(set(days[b]))
    st = dict(nA=nA, nB=nB, daysA=dA, daysB=dB)
    if nA < B_MIN_N or nB < B_MIN_N or dA < B_MIN_DAYS or dB < B_MIN_DAYS:
        return dict(st, status="COLLECTING")
    eff = eff1(y, a, b)
    lo, hi = ci95(boot_eff(y, a, b, H[cluster_col].to_numpy(), rng))
    lp = loo_min_ratio(y, a, b, H.pair.to_numpy(), eff)
    ld = loo_min_ratio(y, a, b, days, eff)
    en = eff1(net, a, b)
    if net_verdict(en, need_sign) == "unpriced":                 # a data gap is not a verdict: not frozen, re-read next week
        return dict(st, status="UNTESTABLE", why=["NET unpriced on a side"])
    why = []
    if _sign(eff) != need_sign:
        why.append("wrong sign")
    if ratio_to_design and not (isinstance(design_eff, (int, float)) and design_eff == design_eff and design_eff != 0
                                and eff / design_eff >= 0.5):
        why.append("< 50 % of DESIGN")
    if not (lo == lo) or not ((lo > 0 and need_sign > 0) or (hi < 0 and need_sign < 0)):
        why.append("CI includes 0")
    if not (lp == lp and lp > 0.5) or not (ld == ld and ld > 0.5):
        why.append("one pair / day carries ≥ 50 %")
    if not (lp == lp and lp >= 0.5) or not (ld == ld and ld >= 0.5):
        why.append("leave-one-out")
    if _sign(en) != need_sign:
        why.append("NET disagrees")
    st.update(eff=eff, ci=(lo, hi), loo_pair=lp, loo_day=ld, net_eff=en, why=sorted(set(why)))
    if "wrong sign" in why or "NET disagrees" in why:
        return dict(st, status="CONTRADICTED")              # §3 / §6 D: the other way = contradicted
    if why:
        return dict(st, status="FAILED")
    return dict(st, status="PASSED", regime=("one regime" if len(regime_signs(H.iloc[np.flatnonzero(a | b)])) < 2 else ""))


# ───────────────────────────────────────────── Q2 (pure) ─────────────────────────────────────────────
def q2_values(d, line, s2c_cuts=None):
    if line == "S2c":
        return Q2.s2c(d, s2c_cuts or {}).to_numpy(float)
    return num(d, f"q2_{line}")


def a2_p(y, a, b, clusters, direction, rng):
    """A2: the one-sided window-bootstrap p = the share of draws NOT in the predicted direction. Pure given rng."""
    dr = boot_eff(y, a, b, clusters, rng)
    dr = dr[np.isfinite(dr)]
    if not len(dr):
        return math.nan
    return float(np.mean(dr >= 0)) if direction < 0 else float(np.mean(dr <= 0))


def design_q2(cohorts, s5, rng):
    """A2 per Q2 line (and the automatic HOLDOUT-only entries) → (rows, s2c_cuts). cohorts = {sleeve: cohort with the guard}."""
    s2c_cuts = Q2.s2c_cuts(design(cohorts["DISTRIBUTION_SHORT"]))
    rows = []
    for line in Q2_ALL:
        sl = Q2.SLEEVE[line]
        base = dict(line=f"Q2|{line}", q2=line, sleeve=sl, dir=Q2.DIRECTION[line])
        if line in Q2.HOLDOUT_ONLY:
            rows.append(dict(base, status="HOLDOUT-ONLY", note="§4: judged on HOLDOUT only, predicted sign"))
            continue
        d = s5 if line == "S5" else cohorts[sl]
        D = labelled(design(d))
        v = q2_values(D, line, s2c_cuts)
        a, b = v == 1.0, v == 0.0
        days = D.day.to_numpy()
        st = dict(base, nA=int(a.sum()), nB=int(b.sum()), daysA=len(set(days[a])), daysB=len(set(days[b])),
                  design_ids=D.signal_id[a | b].tolist() if "signal_id" in D else [])
        if st["nA"] < A_MIN_N or st["nB"] < A_MIN_N or st["daysA"] < A_MIN_DAYS or st["daysB"] < A_MIN_DAYS:
            rows.append(dict(st, status="UNTESTABLE", note="below the DESIGN floor — never read on HOLDOUT"))
            continue
        y = D.y.to_numpy(float)
        eff, en = eff1(y, a, b), eff1(D.net.to_numpy(float), a, b)
        p = a2_p(y, a, b, D.win.to_numpy(), base["dir"], line_rng(base["line"]))
        st.update(eff=eff, p=p, net_eff=en)
        if _sign(eff) == -base["dir"]:
            rows.append(dict(st, status="CONTRADICTED", note="DESIGN effect against the predicted direction — dropped"))
        elif _sign(eff) == 0:
            rows.append(dict(st, status="FAILED", note="no DESIGN effect"))
        elif net_verdict(en, base["dir"]) == "unpriced":
            rows.append(dict(st, status="UNTESTABLE", note="NET unpriced on a side — never read on HOLDOUT"))
        elif net_verdict(en, base["dir"]) == "contra":
            rows.append(dict(st, status="CONTRADICTED", note="NET points the other way (§3) — dropped"))
        elif p == p and p < A2_P:
            rows.append(dict(st, status="SURVIVED DESIGN"))
        else:
            rows.append(dict(st, status="FAILED", note=f"one-sided window bootstrap p {p:.3f} ≥ {A2_P}"))
    return rows, s2c_cuts


# ───────────────────────────────────────────── Q3 (pure given the datasets) ─────────────────────────────────────────────
def order_fam(key):
    """Addendum 4's family for the ORDER: H7a / b / c are listed under their parent H7. Pure."""
    f = I.fam(key)
    return "H7" if f.startswith("H7") else f


def q3_split(key):
    return HALF_SPLIT_611 if (I.fam(key).startswith("H7") or "|" in str(key)) else HALF_SPLIT


def design_q3(T, Cc, cutoff, rng, n_perm=None):
    """A1-Q3 on the DESIGN trigger rows (Addendum 7's floor family, I.family_null) → (rows, bar, order)."""
    t, c = I.cohort(T, Cc, cutoff)
    t = t[(t.signal_time >= DESIGN_START) & (t.signal_time < HOLDOUT_START)]
    c = c[c.trigger_id.isin(t.trigger_id)]
    keys = sorted(T.trigger.unique(), key=lambda k: (I.fam(k), k))
    fam_keys = I.floor_keys(t, c, [k for k in keys if I.fam(k) not in I.HOLDOUT_ONLY])
    obs, null_max = I.family_null(t, c, fam_keys, rng, n_perm or I.N_PERM)
    bar = float(np.nanpercentile(null_max, 95)) if np.isfinite(null_max).any() else math.nan
    rows = []
    for k in keys:
        tk, ck = t[t.trigger == k], c[c.trigger == k]
        n_el = int(tk.trigger_id.isin(set(ck.trigger_id)).sum())
        base = dict(line=f"Q3|{k}", key=k, family=I.fam(k), n_eligible=n_el, days=int(tk[tk.trigger_id.isin(set(ck.trigger_id))].day.nunique()),
                    design_ids=tk.trigger_id.tolist())
        if I.fam(k) in I.HOLDOUT_ONLY:
            rows.append(dict(base, cell=I.FIXED, status="HOLDOUT-ONLY", note="§3: fixed TP 2 / SL 2 cell, judged on HOLDOUT only"))
            continue
        if k not in fam_keys:
            rows.append(dict(base, cell=None, status="UNTESTABLE", note="below the ≥ 20 / ≥ 5-day floor at the DESIGN screen"))
            continue
        if not np.isfinite(obs[k]).any():
            rows.append(dict(base, cell=None, status="UNTESTABLE", note="no cell with a matched Δ"))
            continue
        j = int(np.nanargmax(obs[k]))
        cell, dlt = I.CELLS[j], float(obs[k][j])
        tm, cm = I.matched(tk, ck)
        hs = []
        for sel in (tm.signal_time < q3_split(k), tm.signal_time >= q3_split(k)):
            ids = set(tm[sel].trigger_id)
            x, z = tm[sel][cell], cm[cm.trigger_id.isin(ids)][cell]
            hs.append(float(x.mean() - z.mean()) if len(x) and len(z) else math.nan)
        st = dict(base, cell=cell, delta=dlt, half_a=hs[0], half_b=hs[1])
        if dlt > bar and all(h == h and h > 0 for h in hs):
            rows.append(dict(st, status="SURVIVED DESIGN"))
        else:
            rows.append(dict(st, status="FAILED", note="Δ ≤ the family bar" if not dlt > bar else "halves disagree"))
    # Addendum 4: the order = eligible BASE openers per day per family over DESIGN (counts only)
    span = max(1e-9, (HOLDOUT_START - DESIGN_START).total_seconds() / 86400)
    span7 = max(1e-9, (HOLDOUT_START - DL611_START).total_seconds() / 86400)
    has_c = set(c.trigger_id)
    base_t = t[~t.trigger.astype(str).str.contains("|", regex=False) & t.trigger_id.isin(has_c)]
    order = (base_t.groupby(base_t.trigger.map(order_fam)).size()
             / pd.Series({f: (span7 if f == "H7" else span) for f in base_t.trigger.map(order_fam).unique()}))
    order = order.sort_values(ascending=False)
    return rows, bar, [dict(family=f, per_day=float(v)) for f, v in order.items()]


def q3_holdout_cohort(T, Cc, cutoff):
    t, c = I.cohort(T, Cc, cutoff)
    t = t[t.signal_time >= HOLDOUT_START].copy()
    c = c[c.trigger_id.isin(t.trigger_id)].copy()
    t["hour"] = t.window15.astype(str) if "window15" in t else t.hour        # B3 is window-clustered (§6): window15 is the unit
    wmap = dict(zip(t.trigger_id, t.hour))
    c["hour"] = c.trigger_id.map(wmap)
    return t, c


def b3_gate(tk, ck, cell, rng, *, tu=None, cu=None, parent=None, floor=True):
    """B1 + B3 (+ B3b for a "|c" variant, + the book_unwalkable robustness) for one trigger key on HOLDOUT. Pure given rng."""
    el = tk[tk.trigger_id.isin(set(ck.trigger_id))]
    st = dict(N=int(len(el)), days=int(el.day.nunique()), windows=int(el.hour.nunique()) if "hour" in el else 0)
    if floor and (st["N"] < Q3_B_N or st["days"] < Q3_B_DAYS):
        return dict(st, status="COLLECTING")
    if not len(el):
        return dict(st, status="FAILED", why=["no rows"])
    x = el[cell].to_numpy(float)
    (m_lo, m_hi), (d_lo, d_hi) = I.hour_boot(el, ck, cell, rng)
    tm, cm = I.matched(el, ck)
    dlt = float(tm[cell].mean() - cm[cell].mean()) if len(tm) and len(cm) else math.nan
    kf = I.kelly_floor(x)
    if not kf == kf and (x < 0).sum() == 0 and (x > 0).sum():     # no losers: Kelly → the Wilson WR floor itself (as h4_frozen_lines)
        kf = I.wilson_lo(int((x > 0).sum()), int(np.isfinite(x).sum()))
    mean = float(np.nanmean(x))
    lp = min((el[el.pair != p][cell].mean() for p in el.pair.unique()), default=math.nan)
    ld = min((el[el.day != dd][cell].mean() for dd in el.day.unique()), default=math.nan)
    sp, sd = I.signed_share(el, "pair", cell), I.signed_share(el, "day", cell)
    st.update(mean=mean, ci=(m_lo, m_hi), delta=dlt, delta_ci=(d_lo, d_hi), kelly_floor=kf, loo_pair_min=lp, loo_day_min=ld,
              top_pair_share=sp, top_day_share=sd)
    why = []
    if not (mean > 0 and m_lo > 0):
        why.append("mean / CI")
    if not (dlt > 0 and d_lo > 0):
        why.append("Δ vs control CI")
    if not (kf == kf and kf > 0):
        why.append("Kelly@floor")
    if not (sp == sp and sp < 0.5 and sd == sd and sd < 0.5):
        why.append("one pair / day carries ≥ 50 %")
    if not (lp >= 0.5 * mean and ld >= 0.5 * mean):
        why.append("leave-one-out")
    if tu is not None:                               # Addendum 1: EVERY pass statistic also holds without book_unwalkable rows
        ru = b3_gate(tu, cu, cell, rng, floor=False)
        st["no_unwalkable"] = {k: ru.get(k) for k in ("N", "mean", "ci", "delta", "delta_ci", "kelly_floor", "why")}
        if ru.get("status") != "PASSED":
            why.append("fails without book_unwalkable rows")
    if parent is not None:
        lo, d3 = variant_vs_parent(el, parent, cell, rng)
        st["vs_parent"], st["vs_parent_ci_lo"] = d3, lo
        if not (lo == lo and lo > 0):
            why.append("does not beat its parent (B3b)")
    st["capacity_median_book_usd_05"] = dict(bid=float(el.get("rb_bid_usd_05", pd.Series(dtype=float)).median()),
                                             ask=float(el.get("rb_ask_usd_05", pd.Series(dtype=float)).median()))
    st["why"] = why
    if why:
        return dict(st, status="FAILED")
    return dict(st, status="PASSED", regime=("one regime" if len(regime_signs(el)) < 2 else ""))


def variant_vs_parent(v, p, cell, rng, n=N_BOOT):
    """B3b: mean(variant) − mean(parent openers with cond_depth_scored) in the variant's cell; CI by resampling windows of the union.
    → (ci_lo, effect). Pure given rng."""
    p = p[_bool(p.cond_depth_scored)] if "cond_depth_scored" in p else p
    u = pd.concat([v.assign(_v=1), p.assign(_v=0)], ignore_index=True)
    y = u[cell].to_numpy(float)
    a, b = u._v.to_numpy() == 1, u._v.to_numpy() == 0
    eff = eff1(y, a, b)
    lo, _ = ci95(boot_eff(y, a, b, u.hour.astype(str).to_numpy(), rng, n))
    return lo, eff


# ───────────────────────────────────────────── Q4 (pure) ─────────────────────────────────────────────
S0 = ("breadth_up_15", "breadth_up_60", "liq_long_15", "liq_short_15", "btc_depth_02_usd", "btc_taker_buy_share_15",
      "funding_mean_pct", "premium_mean_pct")


def q4_cuts(mdays):
    """frozen at the DESIGN run from the DESIGN COMPLETE days' market row (no outcome): the 'sign' cut = 0 where a natural zero exists
    (funding, premium), else the DESIGN median; terciles = the DESIGN days' 1/3 and 2/3 quantiles. Pure."""
    m = mdays[(mdays.complete == True) & (mdays.day < HOLDOUT_START.strftime("%Y-%m-%d"))]  # noqa: E712
    out = {}
    for v in S0:
        x = pd.to_numeric(m.get(f"s0_{v}", pd.Series(dtype=float)), errors="coerce").dropna()
        if not len(x):
            out[v] = dict(sign=0.0 if v in Q4_NATURAL_ZERO else None, lo=None, hi=None, n_days=0)
            continue
        out[v] = dict(sign=0.0 if v in Q4_NATURAL_ZERO else float(x.median()), lo=float(x.quantile(1 / 3)),
                      hi=float(x.quantile(2 / 3)), n_days=int(len(x)))
    return out


def q4_gate(daily, cut, gran, rng, regime_n=2):
    """B1 + B2 in DAY units for one sleeve × S0 variable × granularity. daily = DataFrame(day, x, y (daily WORKED rate), net). Two-
    sided (§4). Pure given rng."""
    d = daily.dropna(subset=["x", "y"])
    if gran == "sign":
        if cut.get("sign") is None:
            return dict(status="UNTESTABLE", why=["no DESIGN cut"])
        a, b = (d.x > cut["sign"]).to_numpy(), (d.x <= cut["sign"]).to_numpy()
    else:
        if cut.get("lo") is None or not cut["lo"] < cut["hi"]:
            return dict(status="UNTESTABLE", why=["no DESIGN cut"])
        a, b = (d.x >= cut["hi"]).to_numpy(), (d.x <= cut["lo"]).to_numpy()
    st = dict(daysA=int(a.sum()), daysB=int(b.sum()))
    if st["daysA"] < Q4_B_DAYS or st["daysB"] < Q4_B_DAYS:
        return dict(st, status="COLLECTING")
    y, net = d.y.to_numpy(float), d.net.to_numpy(float)
    eff = eff1(y, a, b)
    lo, hi = ci95(boot_eff(y, a, b, d.day.to_numpy(), rng))
    ld = loo_min_ratio(y, a, b, d.day.to_numpy(), eff)
    en = eff1(net, a, b)
    if net_verdict(en, _sign(eff)) == "unpriced":
        return dict(st, status="UNTESTABLE", why=["NET unpriced on a side"])
    why = []
    if not (lo == lo) or (lo <= 0 <= hi):
        why.append("CI includes 0")
    if not (ld == ld and ld > 0.5):
        why.append("one day carries ≥ 50 % / leave-one-out")
    if _sign(en) != _sign(eff):
        why.append("NET disagrees")
    st.update(eff=eff, ci=(lo, hi), loo_day=ld, net_eff=en, why=why)
    if "NET disagrees" in why:
        return dict(st, status="CONTRADICTED")
    if why:
        return dict(st, status="FAILED")
    return dict(st, status="PASSED", regime="one regime" if regime_n < 2 else "")      # B4


def sleeve_daily(H, mdays, v, last_day=None):
    """per HOLDOUT complete day: the sleeve's WORKED rate / mean NET (days with ≥ 1 labelled row) + the S0 value; only days that END
    at or before the guard (last_day = the last whole UTC day ≤ cutoff − hold), so no partial day. Pure."""
    if "day" not in mdays or "complete" not in mdays or not len(H):
        return pd.DataFrame(columns=["day", "y", "net", "x"])
    m = mdays[_bool(mdays.complete)]
    g = H.groupby("day").agg(y=("y", "mean"), net=("net", "mean"))
    x = m.set_index("day")[f"s0_{v}"] if f"s0_{v}" in m else pd.Series(dtype=float)
    out = g.join(x.rename("x"), how="inner").reset_index()
    out = out[out.day >= HOLDOUT_START.strftime("%Y-%m-%d")]
    return out[out.day <= last_day] if last_day else out


# ───────────────────────────────────────────── state / freeze ─────────────────────────────────────────────
def freeze(state, line, result, cutoff):
    """read ONCE: the first non-COLLECTING result per line is stored and returned at every later run (never re-judged). Pure."""
    if line in state.get("verdicts", {}):
        return state["verdicts"][line], state
    if result.get("status") in ("COLLECTING", "UNTESTABLE", "STALE"):
        return result, state
    state = json.loads(json.dumps(state))
    state.setdefault("verdicts", {})[line] = dict(_jsonable(result), frozen_at=cutoff)
    return state["verdicts"][line], state


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def committed_clean(path):
    """True when `path` is tracked, present in HEAD and unmodified in the working tree (A3: committed before any HOLDOUT read)."""
    rel = os.path.relpath(path, ROOT)
    ok = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True).returncode == 0  # noqa: E731
    if not (os.path.exists(path) and ok("cat-file", "-e", f"HEAD:{rel}") and ok("diff", "--quiet", "HEAD", "--", rel)):
        return False
    out = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    first = out("log", "--diff-filter=A", "--format=%H", "--", rel).splitlines()
    if not first:
        return False
    return out("rev-parse", f"{first[-1]}:{rel}") == out("rev-parse", f"HEAD:{rel}")     # never edited after the first commit


def in_head(path):
    rel = os.path.relpath(path, ROOT)
    return subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"HEAD:{rel}"], capture_output=True).returncode == 0


def head_json(path):
    """the committed (HEAD) version of a JSON file, or None."""
    rel = os.path.relpath(path, ROOT)
    r = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def design_ran_before():
    """True when a DESIGN-phase weekly report exists on disk or SURVIVORS.json appears anywhere in git history."""
    for f in glob.glob(os.path.join(ROOT, "reports", "ROUTE_B_WEEKLY_SCAN_2*.md")):
        try:
            with open(f) as fh:
                if "(DESIGN — run once)" in fh.readline():
                    return True
        except OSError:
            continue
    r = subprocess.run(["git", "-C", ROOT, "log", "--all", "--oneline", "--", os.path.relpath(SURVIVORS, ROOT)], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def state_problem():
    """read once (§6 B): every verdict frozen in the COMMITTED state file must be on disk unchanged — a lost / reset state file would
    re-judge lines on a larger cohort. None = fine."""
    head = head_json(STATE)
    if not head:
        return None
    try:
        disk = json.load(open(STATE))
    except (OSError, ValueError):
        return "reports/ROUTE_B_WEEKLY_SCAN_state.json is committed but missing / unreadable on disk — restore it"
    for k, v in (head.get("verdicts") or {}).items():
        if (disk.get("verdicts") or {}).get(k) != v:
            return f"the committed frozen verdict of {k} is missing or changed on disk — restore reports/ROUTE_B_WEEKLY_SCAN_state.json"
    return None


def load_meta():
    try:
        with open(os.path.join(DATA, "meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def git_head():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


# ───────────────────────────────────────────── loaders ─────────────────────────────────────────────
def load_all():
    S = pd.read_pickle(os.path.join(DATA, "signals.pkl.gz"))
    T, Cc, meta = I.load()
    mp = os.path.join(DATA, "market_days.csv.gz")
    mdays = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame(columns=["day", "complete"])
    return S, T, Cc, meta, mdays


def cohorts_for(S, cutoff):
    co = {sl: guard(sleeve_cohort(S, sl), cutoff, HOLD_H[sl]) for sl in SLEEVES}
    s5 = guard(s5_cohort(S), cutoff, HOLD_H["EXHAUSTION_SHORT"])
    return co, s5


def exit_hashes(meta):
    """each sleeve's exit hash AS PRICED by the loader (meta['exits']). Pure."""
    ex = meta.get("exits") or {}
    return {sl: CM.spec_hash(ex.get(sl)) for sl in SLEEVES}


def config_hashes():
    """the same hashes from today's trading_config.json, as the loader would store them (json round trip, default=str)."""
    specs = CM.exit_specs(CM.load_thresholds())
    return {sl: CM.spec_hash(json.loads(json.dumps(specs.get(sl), default=str))) for sl in SLEEVES}


def data_problem(meta, now):
    """why the loader datasets must not be used this run (None = fine): built > 24 h before the run, data not reaching 72 h past the
    DESIGN end (the FLATBASE guard), or priced under exits other than today's config (re-run build.py). Pure on meta + config."""
    try:
        run = pd.Timestamp(meta["run_utc"])
        end = pd.Timestamp(meta["data_end"])
    except (KeyError, ValueError):
        return "meta.json has no run_utc / data_end"
    if now - run > pd.Timedelta(hours=24):
        return f"the loader ran at {run:%Y-%m-%d %H:%M} UTC (> 24 h ago) — run build.py first"
    if end < HOLDOUT_START + pd.Timedelta(hours=72):
        return f"data end {end:%Y-%m-%d %H:%M} UTC is < 72 h past the DESIGN end — the FLATBASE guard cannot close DESIGN"
    if exit_hashes(meta) != config_hashes():
        return "the datasets were priced under exits other than today's trading_config.json — re-run build.py"
    return None


# ───────────────────────────────────────────── phases ─────────────────────────────────────────────
def dry_run():
    """COUNTS ONLY: cohort sizes, labelled counts, variables, test families, Q3 floor keys. No outcome value is read or printed."""
    S, T, Cc, meta, mdays = load_all()
    cutoff = pd.Timestamp(meta["data_end"])
    co, s5 = cohorts_for(S, cutoff)
    print(f"data to {cutoff} · run {meta.get('run_utc')} · COUNTS ONLY")
    for sl in SLEEVES:
        d = co[sl]
        D = labelled(design(d))
        vars_, skip = screen_vars(design(d))
        defs, A, B = make_tests(D, vars_) if len(D) else ([], np.zeros((0, 0)), np.zeros((0, 0)))
        fl = floor_mask(A, B, D.day.to_numpy()) if len(defs) else np.zeros(0, bool)
        print(f"{SHORT[sl]:9s} cohort DESIGN {len(design(d))} (labelled {len(D)}) · HOLDOUT {len(holdout(d))} · vars {len(vars_)} "
              f"(skipped {len(skip)}) · tests {len(defs)} · reach the floor {int(fl.sum())}")
    print(f"S5 (DL 630 no-spot) DESIGN {len(design(s5))} (labelled {len(labelled(design(s5)))}) · HOLDOUT {len(holdout(s5))}")
    t, c = I.cohort(T, Cc, cutoff)
    t = t[t.signal_time < HOLDOUT_START]
    keys = sorted(T.trigger.unique(), key=lambda k: (I.fam(k), k))
    print("Q3 keys at the floor now:", I.floor_keys(t, c[c.trigger_id.isin(t.trigger_id)],
                                                     [k for k in keys if I.fam(k) not in I.HOLDOUT_ONLY]))
    print(f"Q4 complete DESIGN days: {int(((mdays.complete == True) & (mdays.day < '2026-12-07')).sum())}")  # noqa: E712


def run_design(now):
    S, T, Cc, meta, mdays = load_all()
    cutoff = pd.Timestamp(meta["data_end"])
    co, s5 = cohorts_for(S, cutoff)
    q1 = {SHORT[sl]: design_q1(co[sl], sl, np.random.default_rng(SEED)) for sl in SLEEVES}     # one fresh seed per family
    q2, s2c_cuts = design_q2(co, s5, None)                                                       # per-line seeds inside
    q3, q3_bar, order = design_q3(T, Cc, cutoff, np.random.default_rng(SEED))
    surv = dict(prereg="reports/ROUTE_B_WEEKLY_SCAN_PREREG.md (+ Addenda 1–10)", run_utc=now.isoformat(timespec="seconds"),
                data_end=str(cutoff), git_head=git_head(), seed=SEED, n_rotations=N_ROT, exit_hash=exit_hashes(meta),
                q1=q1, q2=q2, s2c_cuts=s2c_cuts, q3=dict(bar=q3_bar, keys=q3, addendum4_order=order),
                q4_cuts=q4_cuts(mdays),
                willy="UNTESTABLE — the loader cannot price plain FRENZY_WILLY's V2b exit (§1)")
    with open(SURVIVORS, "w") as f:
        json.dump(_jsonable(surv), f, indent=1)
    return surv, S, co, s5, mdays, meta


def recheck_stale(line, sv, co, s5, s2c_cuts, state, hashes):
    """§2 'Exit changes': a not-yet-judged Q1 / Q2 survivor whose sleeve exit changed is re-checked ONCE per change on the re-priced
    DESIGN rows (the frozen ids), with the frozen cut / bar. → (status, state)."""
    sl = sv["sleeve"].split(":")[0]
    seen = state.get("recheck", {}).get(line, {}).get("hash") or sv.get("_hash0")
    if hashes.get(sl) == seen:
        return None, state
    d = s5 if sv.get("q2") == "S5" else co[sl]
    D = labelled(design(d))
    D = D[D.signal_id.isin(set(sv.get("design_ids") or []))].reset_index(drop=True)
    if not len(D):
        return "STALE", state                  # the frozen DESIGN rows are not visible this run → waits, nothing recorded
    if sv["line"].startswith("Q1|"):
        X = {v: num(D, v) for v in (sv["var"], sv.get("var2")) if v}
        a, b = sides(X, sv)
        eff, t = cluster_t(D.y.to_numpy(float), a[:, None], b[:, None], onehot(D.win.to_numpy()))
        ck = a1_checks(D, a, b, eff[0], is_dl611(sv))
        fl = floor_mask(a[:, None], b[:, None], D.day.to_numpy())[0]
        ok = bool(fl and abs(t[0]) > sv["_bar"] and ck["halves_ok"] and net_verdict(ck["net_eff"], _sign(eff[0])) == "ok")
    else:
        v = q2_values(D, sv["q2"], s2c_cuts)
        a, b = v == 1.0, v == 0.0
        y = D.y.to_numpy(float)
        eff = eff1(y, a, b)
        fl = floor_mask(a[:, None], b[:, None], D.day.to_numpy())[0]
        p = a2_p(y, a, b, D.win.to_numpy(), sv["dir"], line_rng(line))
        ok = bool(fl and _sign(eff) == sv["dir"] and net_verdict(eff1(D.net.to_numpy(float), a, b), sv["dir"]) == "ok"
                  and p == p and p < A2_P)
    state = json.loads(json.dumps(state))
    state.setdefault("recheck", {})[line] = dict(hash=hashes.get(sl), ok=ok)
    return ("SURVIVED DESIGN" if ok else "FAILED"), state


def stale_result(pre, state, line, cutoff):
    """the outcome of an exit re-check: FAILED is frozen (dropped for good) · STALE (rows not visible) waits · else None. Pure."""
    if pre == "FAILED":
        return freeze(state, line, dict(status="FAILED", why=["exit re-check (STALE) failed"]), cutoff)
    if pre == "STALE":
        return dict(status="STALE", why=["exit changed; the frozen DESIGN rows are not visible this run"]), state
    return None, state


def run_holdout(surv, now):
    S, T, Cc, meta, mdays = load_all()
    cutoff = pd.Timestamp(meta["data_end"])
    co, s5 = cohorts_for(S, cutoff)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    hashes = exit_hashes(meta)
    lines = []
    # Q1 survivors
    for sh, q in surv["q1"].items():
        sl = next(k for k, v in SHORT.items() if v == sh)
        H = labelled(holdout(co[sl]))
        for sv in q.get("survivors", []):
            sv = dict(sv, sleeve=sl, _bar=q["bar"], _hash0=surv["exit_hash"].get(sl), design_ids=q.get("design_ids"))
            res = None
            if sv["line"] not in state.get("verdicts", {}):
                pre, state = recheck_stale(sv["line"], sv, co, s5, surv.get("s2c_cuts"), state, hashes)
                res, state = stale_result(pre, state, sv["line"], str(cutoff))
            if res is None:
                X = {v: num(H, v) for v in (sv["var"], sv.get("var2")) if v}
                a, b = sides(X, sv) if len(H) else (np.zeros(0, bool), np.zeros(0, bool))
                res = b_gate(H, a, b, sv["eff"], _sign(sv["eff"]), line_rng(sv["line"]), ratio_to_design=not sv["holdout_only"])
                res, state = freeze(state, sv["line"], res, str(cutoff))
            lines.append(dict(sv, holdout=res))
        lines += [dict(x, sleeve=sl) for x in q.get("dropped", [])]
    # Q2 lines
    for sv in surv["q2"]:
        if sv["status"] not in ("SURVIVED DESIGN", "HOLDOUT-ONLY"):
            lines.append(dict(sv, holdout=dict(status=sv["status"], why=[sv.get("note", "")])))
            continue
        d = s5 if sv["q2"] == "S5" else co[sv["sleeve"]]
        H = labelled(holdout(d))
        res = None
        if sv["status"] == "SURVIVED DESIGN" and sv["line"] not in state.get("verdicts", {}):
            pre, state = recheck_stale(sv["line"], dict(sv, _hash0=surv["exit_hash"].get(sv["sleeve"].split(":")[0])), co, s5,
                                       surv.get("s2c_cuts"), state, hashes)
            res, state = stale_result(pre, state, sv["line"], str(cutoff))
        if res is None:
            v = q2_values(H, sv["q2"], surv.get("s2c_cuts"))
            res = b_gate(H, v == 1.0, v == 0.0, sv.get("eff"), sv["dir"], line_rng(sv["line"]),
                         ratio_to_design=sv["status"] == "SURVIVED DESIGN")
            res, state = freeze(state, sv["line"], res, str(cutoff))
        lines.append(dict(sv, holdout=res))
    # Q3 keys
    th, ch = q3_holdout_cohort(T, Cc, cutoff)
    tu, cu = I.cohort(T, Cc, cutoff, drop_unwalkable=True)
    tu = tu[tu.signal_time >= HOLDOUT_START].copy()
    cu = cu[cu.trigger_id.isin(tu.trigger_id)].copy()
    tu["hour"] = tu.window15.astype(str) if "window15" in tu else tu.hour
    cu["hour"] = cu.trigger_id.map(dict(zip(tu.trigger_id, tu.hour)))
    for sv in surv["q3"]["keys"]:
        if sv["status"] not in ("SURVIVED DESIGN", "HOLDOUT-ONLY"):
            lines.append(dict(sv, holdout=dict(status=sv["status"], why=[sv.get("note", "")])))
            continue
        k, cell = sv["key"], sv["cell"]
        pk = k.split("|")[0]
        par = th[(th.trigger == pk) & th.trigger_id.isin(set(ch[ch.trigger == pk].trigger_id))] if "|" in k else None   # eligible, like the variant
        res = b3_gate(th[th.trigger == k], ch[ch.trigger == k], cell, line_rng(sv["line"]), tu=tu[tu.trigger == k], cu=cu[cu.trigger == k],
                      parent=par)
        res, state = freeze(state, sv["line"], res, str(cutoff))
        lines.append(dict(sv, holdout=res))
    # Q4 market lines (HOLDOUT-only, day units)
    for sl in SLEEVES:
        H = labelled(holdout(co[sl]))
        last_day = ((cutoff - pd.Timedelta(hours=HOLD_H[sl])).floor("D") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for v, cut in surv["q4_cuts"].items():
            daily = sleeve_daily(H, mdays, v, last_day)
            reg = len(regime_signs(H[H.day.isin(set(daily.day))]))
            for gran in ("sign", "tercile"):
                ln = f"Q4|{SHORT[sl]}|{v}|{gran}"
                res, state = freeze(state, ln, q4_gate(daily, cut, gran, line_rng(ln), regime_n=reg), str(cutoff))
                lines.append(dict(line=ln, sleeve=sl, var=v, gran=gran, status="HOLDOUT-ONLY", holdout=res))
    json.dump(_jsonable(state), open(STATE, "w"), indent=1)
    return lines, S, co, s5, mdays, meta


# ───────────────────────────────────────────── report ─────────────────────────────────────────────
def f2(x, p=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (list, tuple)):
        return f"{f2(x[0], p, pct)} … {f2(x[1], p, pct)}"
    return f"{x * 100:+.{max(p - 2, 0)}f} pp" if pct else f"{x:+.{p}f}"


def final_status(ln):
    h = ln.get("holdout") or {}
    st = h.get("status") or ln.get("status")
    if st in ("HOLDOUT-ONLY", "COLLECTING") and ln.get("status") == "HOLDOUT-ONLY":
        return "COLLECTING", "HOLDOUT-only line"
    if st == "COLLECTING" and ln.get("status") in ("SURVIVED DESIGN",):
        return "SURVIVED DESIGN", "HOLDOUT collecting"
    if st == "COLLECTING":
        return "COLLECTING", ""
    return st, h.get("regime", "")


def disclosure():
    try:
        txt = open(PREREG).read()
        a = txt.index("## 0 ·")
        b = txt.index("\n---", a)
        sec = txt[a:b].strip().replace("## 0 ·", "## §0 ·", 1)
    except (OSError, ValueError):
        sec = "## §0 · disclosure — PREREG not readable"
    looks = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "reports", "ROUTE_B_INTERIM_Q3_*.md")))
    sec += ("\n\nInterim Q3 looks (Addendum 6 — every H outcome up to each cut-off counts as SEEN): "
            + (", ".join(looks) if looks else "none") + ". H4 study: reports/H4_ASK_VACUUM_STUDY_2026-11-24.md (Addenda 8–9).")
    return sec


def coverage_mark(d):
    if not len(d):
        return "⚪"
    s = float(_bool(d.get("rb_has_book", pd.Series(False, index=d.index))).mean())
    return "🟢" if s >= 0.9 else ("🟡" if s >= 0.5 else "🔴")


def sleeve_descriptives(H):
    """BTC-sign split + path decomposition + Q5 capacity on a sleeve's HOLDOUT cohort — aggregates only (Addendum 2)."""
    L = []
    if not len(H):
        return ["_no HOLDOUT rows yet_"]
    lab = H[H.y.notna()]
    sg = pd.to_numeric(lab.get("btc_d_trend"), errors="coerce").map(lambda v: "unscored" if v != v else ("bull" if v > 0 else ("bear" if v < 0 else "tie")))
    t = lab.groupby(sg).agg(N=("y", "size"), worked=("y", "mean"), net=("net", "mean"), days=("day", "nunique"))
    L += ["**BTC daily-trend split (descriptive)**", "", "| BTC 1 d EMA50 vs EMA200 | N | days | WORKED | avg NET % |", "|---|---|---|---|---|"]
    L += [f"| {k} | {int(r.N)} | {int(r.days)} | {r.worked * 100:.0f} % | {f2(r.net)} |" for k, r in t.iterrows()]
    if "out_lbl_group" in H:
        g = H.groupby(H.out_lbl_group.fillna("PENDING")).agg(N=("net", "size"), net=("net", "sum"))
        L += ["", "**Path decomposition (entry × exit ①, descriptive)**", "", "| group | N | Σ NET % |", "|---|---|---|"]
        L += [f"| {k} | {int(r.N)} | {f2(r.net)} |" for k, r in g.iterrows()]
    k = H[_bool(H.get("today_filters_pass", pd.Series(False, index=H.index)).fillna(False))]
    if len(k):
        L += ["", f"**Q5 · cost and capacity — rows today's stack keeps (N {len(k)}; summaries only)**", "",
              "| item | median |", "|---|---|"]
        for c, lab_ in (("out_entry_slip_pct", "entry slip % (loader)"), ("out_exit_slip_pct", "exit slip % (loader)"),
                        ("journal_model_slip_pct", "journal model slip %"), ("fb_cost_sell_6000", "full-book $6k sell cost (bps)"),
                        ("fb_cost_buy_6000", "full-book $6k buy cost (bps)"), ("rb_bid_usd_05", "bid $ within 0.5 % (20-level)"),
                        ("rb_ask_usd_05", "ask $ within 0.5 % (20-level)"), ("fb_bid_usd_05", "bid $ within 0.5 % (full book)"),
                        ("fb_ask_usd_05", "ask $ within 0.5 % (full book)")):
            x = pd.to_numeric(k.get(c), errors="coerce").dropna() if c in k else pd.Series(dtype=float)
            L.append(f"| {lab_} | {('%.3f' % x.median()) if len(x) else '–'} (n {len(x)}) |")
        if "out_cost_flag" in k:
            L.append(f"| cost source | {', '.join(f'{a} {b}' for a, b in k.out_cost_flag.fillna('').replace('', 'unpriced').value_counts().items())} |")
    return L


def md(x):
    """a markdown table cell: escape the pipes that line ids / keys carry (GFM splits cells on them, even inside code spans). Pure."""
    return str(x).replace("|", "\\|")


def haircut_text(de, unit):
    """B5: a DESIGN Δ quoted after the 30–50 % in-sample haircut, as the kept 50–70 % range (WORKED effects in pp, Q3 Δ in net %). Pure."""
    if not isinstance(de, (int, float)) or not de == de:
        return "–"
    k = 100.0 if unit == "pp" else 1.0
    a, b = sorted((de * 0.5 * k, de * 0.7 * k))
    return f"{a:+.1f} … {b:+.1f} {unit}"


def h4_colour(v):
    w = str(v).split(" ")[0]
    return COLOUR.get({"PASS": "PASSED", "FAIL": "FAILED"}.get(w, w), "⚪")


def status_row(ln):
    st, note = final_status(ln)
    h = ln.get("holdout") or {}
    de = ln.get("eff", ln.get("delta"))
    hair = haircut_text(de, "pp" if "eff" in ln else "%")
    n = (f"{h.get('nA')}/{h.get('nB')} · {h.get('daysA')}/{h.get('daysB')} d" if "nA" in h else
         (f"{h.get('N')} · {h.get('days')} d" if "N" in h else (f"{h.get('daysA')}/{h.get('daysB')} d" if "daysA" in h else "–")))
    eff = h.get("eff", h.get("mean"))
    ci = h.get("ci")
    why = "; ".join(x for x in (h.get("why") or [ln.get("note")]) if x)
    return (f"| {COLOUR.get(st, '⚪')} {st}{(' · ' + note) if note else ''} | `{md(ln['line'])}` | {hair} | {n} | {f2(eff, 3)} | "
            f"{f2(ci, 3) if ci else '–'} | {why or '–'} |")


def write_report(now, phase, surv, lines, S, co, s5, mdays, meta, h4=None, extra=None, charts=None):
    date = now.strftime("%Y-%m-%d")
    p = os.path.join(ROOT, "reports", f"ROUTE_B_WEEKLY_SCAN_{date}{'_DESIGN' if phase.startswith('DESIGN') else ''}.md")
    L = [f"# 📡 Route B weekly research scan — {date} ({phase})", "",
         f"Data to {meta.get('data_end')} (loader run {meta.get('run_utc')}) · DESIGN {DESIGN_START:%m-%d %H:%M} → {HOLDOUT_START:%m-%d} · "
         f"HOLDOUT ≥ {HOLDOUT_START:%Y-%m-%d} · SURVIVORS frozen {surv.get('run_utc')} at {surv.get('git_head')} · research only, alert "
         "only — nothing arms. Summaries only (Addendum 2).", ""]
    if extra:
        L += extra + [""]
    L += [disclosure(), "", "## Coverage and blind spots", "",
          "| sleeve | mark (book at entry) | DESIGN cohort (labelled) | HOLDOUT cohort (labelled) | Q1 variables | own-read (< 50 % scored) | not testable |",
          "|---|---|---|---|---|---|---|"]
    for sl in SLEEVES:
        q = surv["q1"][SHORT[sl]]
        d = co[sl]
        L.append(f"| {SHORT[sl]} | {coverage_mark(d)} | {len(design(d))} ({len(labelled(design(d)))}) | {len(holdout(d))} "
                 f"({len(labelled(holdout(d)))}) | {q['n_vars']} | {len(q['low_coverage'])} | {len(q['skipped'])} |")
    L.append(f"| no-spot (S5) | {coverage_mark(s5)} | {len(design(s5))} ({len(labelled(design(s5)))}) | {len(holdout(s5))} "
             f"({len(labelled(holdout(s5)))}) | – | – | – |")
    L += ["", "Blind spots: plain FRENZY_WILLY is **UNTESTABLE** (V2b exit not priceable — §1); market (mk_*) columns are Q4 only; "
          "variables never scored or constant are listed per sleeve in SURVIVORS.json (`skipped`). Low-coverage variables are screened on "
          "their scored rows only (they stay in the family).", ""]
    L += ["## Status (every line)", "",
          "| status | line | DESIGN effect after the 30–50 % haircut | HOLDOUT N (sides · days) | HOLDOUT effect | CI 95 % | why |",
          "|---|---|---|---|---|---|---|"]
    for sl in SLEEVES:
        q = surv["q1"][SHORT[sl]]
        L.append(f"| ⚪ screen | `Q1\\|{SHORT[sl]}` family: {q['n_family']} tests at the floor of {q['n_tests']} · bar max \\|t\\| {f2(q.get('bar'))} "
                 f"· {len(q['survivors'])} survived DESIGN{(' · ' + q['note']) if q.get('note') else ''} | – | – | – | – | – |")
    order = {r["family"]: i for i, r in enumerate(surv["q3"].get("addendum4_order", []))}

    def sec_key(ln):
        s = ln["line"].split("|")
        if s[0] == "Q3":
            return (5, order.get(order_fam(ln["key"]), 99), ln["key"])
        sh = s[1] if s[0] in ("Q1", "Q4") else SHORT.get(str(ln.get("sleeve")).split(":")[0], "")
        if ln.get("q2") == "S5":
            return (4, 0, ln["line"])
        if s[0] == "Q4":
            return (6, list(SHORT.values()).index(sh), ln["line"])
        return (list(SHORT.values()).index(sh) if sh in SHORT.values() else 9, 0, ln["line"])
    q4c = {}
    for ln in sorted(lines, key=sec_key):
        if ln["line"].startswith("Q4|") and final_status(ln)[0] == "COLLECTING":
            h = ln.get("holdout") or {}
            sh = ln["line"].split("|")[1]
            q4c.setdefault(sh, []).append(min(h.get("daysA", 0) or 0, h.get("daysB", 0) or 0))
            continue
        L.append(status_row(ln))
    for sh, v in q4c.items():
        L.append(f"| ⚪ COLLECTING | `Q4\\|{sh}` — {len(v)} market lines (8 S0 variables × sign / tercile) | – | fewest days on the "
                 f"smaller side: {min(v)} (needs {Q4_B_DAYS}) | – | – | – |")
    if h4:
        L += [f"| {h4_colour(v)} {v} | `H4 frozen {k}` (Addendum 9) | – | – | – | – | – |"
              for k, v in h4.items()]
    L += ["| ⚪ UNTESTABLE | `FRENZY_WILLY` (plain) | – | – | – | – | V2b exit not priceable |", ""]
    hold_ok = phase.startswith("HOLDOUT")              # A3: no HOLDOUT outcome is shown before SURVIVORS.json is committed
    for sl in SLEEVES:
        L += [f"## {SHORT[sl]}", ""]
        L += (sleeve_descriptives(holdout(co[sl])) if hold_ok else ["_HOLDOUT descriptives are not read until SURVIVORS.json is committed (A3)._"])
        L += [""]
    L += ["## no-spot (S5, DL 630 cohort)", "", f"HOLDOUT rows {len(holdout(s5))} (labelled {len(labelled(holdout(s5)))}).", ""]
    L += ["## Triggers (Q3, Addendum 4 order)", "", f"Family bar (DESIGN, Addendum 7): Δ {f2(surv['q3'].get('bar'))} net % · order (eligible "
          "base openers per day over DESIGN, frozen): " + ", ".join(f"{r['family']} {r['per_day']:.1f}/day" for r in surv["q3"].get("addendum4_order", [])),
          "", "| key | status | cell | DESIGN eligible N · days | HOLDOUT eligible N · days · separate windows |", "|---|---|---|---|---|"]
    byk = {ln["key"]: ln for ln in lines if ln["line"].startswith("Q3|")}
    for r in sorted(surv["q3"]["keys"], key=lambda r: (order.get(order_fam(r["key"]), 99), r["key"])):
        ln = byk.get(r["key"], r)
        h = ln.get("holdout") or {}
        st, note = final_status(ln)
        L.append(f"| {md(r['key'])} | {COLOUR.get(st, '⚪')} {st}{(' · ' + note) if note else ''} | {str(r.get('cell') or '–').replace('g_', '').replace('_net', '')} | "
                 f"{r.get('n_eligible')} · {r.get('days')} | {h.get('N', '–')} · {h.get('days', '–')} · {h.get('windows', '–')} |")
    L += [""]
    charts = [c for c in (charts or []) if c]
    if charts:
        L += ["## Event studies (PASSED lines — aggregate paths)", ""] + [f"![{os.path.basename(c)}]({os.path.basename(c)})" for c in charts] + [""]
    L += ["## Market (Q4, day units, HOLDOUT only)", "", f"Complete HOLDOUT days: "
          f"{int(((mdays.complete == True) & (mdays.day >= HOLDOUT_START.strftime('%Y-%m-%d'))).sum())}. Lines: 4 sleeves × 8 S0 variables × "  # noqa: E712
          "2 granularities, each two-sided with no correction across lines (as frozen) — about 3 false CI exclusions are expected by "
          "chance, so a Q4 pass is only ever a sleeve on/off OBSERVE line.", ""]
    open(p, "w").write("\n".join(L) + "\n")
    return p


def fav_path(k, entry_ms, side, n_min):
    """favourable % path (closes) for n_min minutes from the open of the entry minute; None unless the bars are complete. Pure."""
    if k is None or not len(k):
        return None
    k = k[k.open_ms >= entry_ms].iloc[:n_min]
    if len(k) < n_min or int(k.open_ms.iloc[0]) != int(entry_ms):
        return None
    e = float(k.o.iloc[0])
    return side * (k.c.to_numpy(float) / e - 1) * 100


def band(paths):
    """median and interquartile band per minute over equal-length paths. Pure."""
    P = np.vstack(paths)
    return np.median(P, 0), np.percentile(P, 25, 0), np.percentile(P, 75, 0)


def event_chart_for(ln, co, s5, now, s2c_cuts=None, cap=150):
    """§7 event-study chart for a PASSED line: median (IQR) favourable path of side A vs side B (Q1 / Q2), or the triggers vs their
    matched random minutes (Q3), over the hold. Network (klines); a failure returns None and the report says nothing is drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import sources as SRC
        now_ms = int(now.value // 1_000_000)
        groups = {}
        if ln["line"].startswith("Q4|"):
            return None                                   # a day-unit market line has no entry path; its day table is the read
        if ln["line"].startswith("Q3|"):
            _, T, Cc, meta, _ = load_all()
            th, ch = q3_holdout_cohort(T, Cc, pd.Timestamp(meta["data_end"]))
            tk = th[th.trigger == ln["key"]]
            ck = ch[ch.trigger_id.isin(set(tk.trigger_id))]
            n_min = 240
            for lab, d in (("trigger", tk), ("random minutes", ck)):
                ps = []
                for r in d.head(cap).itertuples():
                    sd = 1 if str(getattr(r, "direction", "LONG")).upper() == "LONG" else -1
                    e = int(r.h_entry_ms)
                    ps.append(fav_path(SRC.klines(r.pair, e, e + (n_min + 1) * CM.MIN_MS, now_ms), e, sd, n_min))
                groups[lab] = [p for p in ps if p is not None]
        else:
            sl = ln["sleeve"].split(":")[0]
            d = s5 if ln.get("q2") == "S5" else co[sl]
            H = labelled(holdout(d))
            if ln["line"].startswith("Q1|"):
                a, b = sides({v: num(H, v) for v in (ln["var"], ln.get("var2")) if v}, ln)
            else:
                v = q2_values(H, ln["q2"], s2c_cuts)
                a, b = v == 1.0, v == 0.0
            sd = 1 if sl == "FRENZY_WILLY_HT" else -1
            n_min = HOLD_H[sl] * 60
            for lab, m in (("side A", a), ("side B", b)):
                ps = []
                for r in H[m].head(cap).itertuples():
                    e = getattr(r, "out_entry_ms", None)
                    if e is None or e != e:
                        continue
                    e = int(e) // CM.MIN_MS * CM.MIN_MS
                    ps.append(fav_path(SRC.klines(r.pair, e, e + (n_min + 1) * CM.MIN_MS, now_ms), e, sd, n_min))
                groups[lab] = [p for p in ps if p is not None]
        fig, ax = plt.subplots(figsize=(8, 4))
        for lab, ps in groups.items():
            if ps:
                m, lo, hi = band(ps)
                x = np.arange(1, len(m) + 1)
                ax.plot(x, m, label=f"{lab} (n {len(ps)})")
                ax.fill_between(x, lo, hi, alpha=0.2)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("minutes from entry")
        ax.set_ylabel("favourable % (median, IQR)")
        ax.set_title(ln["line"], fontsize=9)
        ax.legend()
        slug = "".join(ch if ch.isalnum() else "_" for ch in ln["line"])[:80]
        out = os.path.join(ROOT, "reports", f"ROUTE_B_WEEKLY_SCAN_{now:%Y-%m-%d}_{slug}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception:  # noqa: BLE001 — a chart never blocks the report
        return None


def h4_status():
    """the H4 frozen lines (Addendum 9): run their script; the status shown is the frozen verdict, else COLLECTING."""
    try:
        import h4_frozen_lines as H4
        H4.main()
    except Exception as e:  # noqa: BLE001 — the scan must still report; the failure is shown
        return {"H4-T": f"UNTESTABLE (run failed: {type(e).__name__})", "H4-B": "UNTESTABLE"}
    st = json.load(open(H4_STATE)) if os.path.exists(H4_STATE) else {}
    return {k: (st[k]["status"] if k in st else "COLLECTING") for k in ("H4-T", "H4-B")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="counts only — any day, never an outcome")
    ap.add_argument("--late-design", action="store_true", help="operator OK to run the one-time DESIGN screen after 2026-12-10")
    a = ap.parse_args(argv)
    if a.dry_run:
        dry_run()
        return 0
    now = pd.Timestamp.now("UTC")
    if now < FIRST_RUN:
        print(f"Route B weekly scan: the first run is {FIRST_RUN:%Y-%m-%d} (PREREG §7) — nothing read.")
        return 0
    if not os.path.exists(SURVIVORS) and (in_head(SURVIVORS) or design_ran_before()):
        print("refused: the DESIGN screen already ran (SURVIVORS.json in git history or a DESIGN report on disk) but SURVIVORS.json is "
              "missing — restore it (`git checkout -- reports/ROUTE_B_WEEKLY_SCAN_SURVIVORS.json`); the DESIGN screen never runs twice.")
        return 2
    prob = data_problem(load_meta(), now)
    if prob:
        print(f"refused: {prob}. Nothing read.")
        return 2
    if not os.path.exists(SURVIVORS) and now >= FIRST_RUN + pd.Timedelta(days=1) and not a.late_design:
        print("refused: the DESIGN screen is due on 2026-12-10 (§2: DESIGN = rows visible at the 12-10 run). Running it later takes in "
              "late rows — only with the operator's OK, as `--late-design` (the report then says so).")
        return 2
    if not os.path.exists(SURVIVORS):
        surv, S, co, s5, mdays, meta = run_design(now)
        n = sum(len(q["survivors"]) for q in surv["q1"].values())
        late = f" **LATE: run {now:%Y-%m-%d %H:%M} UTC with --late-design (operator OK) — rows visible now, not at 12-10.**" if a.late_design else ""
        extra = ["> **DESIGN screen run once.**" + late + " SURVIVORS.json written. **HOLDOUT is NOT read until the operator commits "
                 "`reports/ROUTE_B_WEEKLY_SCAN_SURVIVORS.json` (A3)** — the next weekly run reads it."]
        lines = [dict(sv, sleeve=next(k for k, v in SHORT.items() if v == sh)) for sh, q in surv["q1"].items() for sv in q["survivors"]]
        lines += [dict(x, sleeve=next(k for k, v in SHORT.items() if v == sh)) for sh, q in surv["q1"].items() for x in q.get("dropped", [])]
        lines += surv["q2"] + surv["q3"]["keys"]
        p = write_report(now, "DESIGN — run once", surv, lines, S, co, s5, mdays, meta, extra=extra)
        print(f"DESIGN done: {n} Q1 survivors · Q2 {sum(r['status'] == 'SURVIVED DESIGN' for r in surv['q2'])} · Q3 "
              f"{sum(r['status'] == 'SURVIVED DESIGN' for r in surv['q3']['keys'])} → {os.path.relpath(SURVIVORS, ROOT)} "
              f"(COMMIT IT before the next run) · report {os.path.relpath(p, ROOT)}")
        return 0
    if not committed_clean(SURVIVORS):
        print("HOLDOUT refused: reports/ROUTE_B_WEEKLY_SCAN_SURVIVORS.json is not committed (or was modified) — A3. Nothing read.")
        return 2
    sp = state_problem()
    if sp:
        print(f"HOLDOUT refused: {sp}. Nothing read.")
        return 2
    surv = json.load(open(SURVIVORS))
    lines, S, co, s5, mdays, meta = run_holdout(surv, now)
    h4 = h4_status()
    charts = [event_chart_for(ln, co, s5, now, surv.get("s2c_cuts")) for ln in lines if (ln.get("holdout") or {}).get("status") == "PASSED"]
    p = write_report(now, "HOLDOUT status", surv, lines, S, co, s5, mdays, meta, h4=h4, charts=charts)
    print(f"report → {os.path.relpath(p, ROOT)} · commit reports/ROUTE_B_WEEKLY_SCAN_state.json with it (the frozen verdicts live there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
