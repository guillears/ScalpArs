"""🎲 Best-case LUCK TEST per watch line (DECISION_LOG 473, operator 2026-11-05: "why not doing for every rule in watchlist?" → "yes build it").

The question: a line was FOUND by searching the trades it is now judged on (in-sample). How often does pure luck — the same search run on
results that carry no information — produce a line at least this good? Only the BUILT-ON trades are used (every master fill of the sleeve on
today's stack + current-batch fills opened before the line's freeze); the clean trades after the freeze are the forward test and stay out.

BLOCK lines (they claim the blocked fills do worse): Δ = mean(rest) − mean(zone), booked % on today's exit, scored as t = Δ ÷ (sd ·
√(1/n_zone + 1/n_rest)) with the cohort's sd — a STANDARDISED gap, so a 5-fill cell needs a far bigger Δ than a 28-fill zone (a raw-Δ max
over thousands of tiny cells beats every real line by construction: the first 11-05 run read p 1.0 on every line). The null: results
shuffled in whole UTC-DAY blocks (same-day fills move together — the WINDOW-UNITS rule), then the SEARCH re-run: the best Δ over the family
(every numeric entry_* column scored on ≥ 80 % of the fills: sign / median / both terciles, each side as the zone · every column pair's
four median quadrants · the line's own zone), zone and rest ≥ MIN_SIDE scored fills each. Unscored fills are in neither side of a candidate
(unscored ≠ rest). p(after search) = (1 + #shuffles whose best ≥ the line's Δ) / (1 + R); p(as a single rule) = the same with only the
line's own zone (no search) — the gap between them is what the search cost.
EXIT / TIMING lines (FB_CONFIRM, FAN flips TP +2: they claim a gain per trade): Δ per trade vs today's exit; null = each day's Δs flip sign
together (no effect = as likely +x as −x), best mean over the alternatives that were tried and are priced (FAN TP 1.5 / 2 / 3 · FB_CONFIRM at
+5 / −10 and +1 / −4 · waits of 1 / 2 / 4 / 8 / 12 / 24 h · selling 1–7 % higher: the alternatives its registration study tried, priced in
fb_combo.py since DECISION_LOG 474 — the operator caught that a 2-alternative penalty made a short Flatbase history read REAL).
LIMITS (deep review 473): the family is the standard sweep, not each line's own registration search — WIDER for some lines (LONG_ADX35 was
one post-hoc 1D cut; a sleeve with more stamped columns gets a bigger family), NARROWER for others (WILLY_GAP searched 28,545 cells;
ML_MKT_HOT 49 rebuilt market conditions that are not entry columns) → p(after search) is not a bound either way; p(1D search) (every 1D
split + the line's own zone) is shown beside it. Categorical / flag columns are not in the family (numeric only); booking and post-fill
columns (POST_FILL) are left out. Day blocks have different sizes, so a shuffled day can spill over two target days (slightly weaker
day clustering → mildly anti-conservative for market-wide splits). The line's OWN zone counts unscored fills as rest (as the best case
applies it); family candidates keep unscored fills out of both sides.
Verdict (frozen): p ≤ 0.05 REAL · ≤ 0.20 UNCLEAR · > 0.20 NOT ABOVE LUCK · built-on N < MIN_N or a side < MIN_SIDE → TOO FEW.
It never moves a gate and never changes the locked 30–50 % in-sample cut; a NOT ABOVE LUCK line's history counts for nothing (the
"luck-checked" totals switch it off), the others keep the cut until their clean checkpoint. PURE core; best_dcr.py calls it.
"""
import numpy as np
import pandas as pd

R = 2000          # deep review 473: at 200 the Monte-Carlo SE near the 0.05 / 0.20 cuts (≈ 0.015 / 0.026) could move a band
SEED = 473
# booking / post-fill entry_* columns (written at or after the fill, never a decision input) — out of the search family
POST_FILL = ("entry_price", "entry_fee", "entry_strategy", "entry_order_type", "entry_slippage_pct", "entry_desired_notional",
             "entry_liquidity_cap_notional", "entry_frenzy_vwap")
POST_FILL_PREFIX = ("entry_bracket_",)
MIN_SIDE = 5
MIN_N = 15
REAL, UNCLEAR, LUCK, FEW = "REAL", "UNCLEAR", "NOT ABOVE LUCK", "TOO FEW"


def verdict(p):
    if p is None or p != p:
        return FEW
    return REAL if p <= 0.05 else UNCLEAR if p <= 0.20 else LUCK


def candidate_cols(df):
    """numeric entry_* columns scored on ≥ 80 % of the fills with > 3 distinct values. PURE."""
    out = []
    for c in sorted(c for c in df.columns if str(c).startswith("entry_") and c not in POST_FILL and not str(c).startswith(POST_FILL_PREFIX)):
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 0.8 * len(df) and x.nunique() > 3:
            out.append(c)
    return out


def family(df, cols, two_d=True):
    """→ (Z, V): n × K zone / scored matrices (bool) over every 1D split (+ every 2D median quadrant when two_d). PURE."""
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float) if cols else np.zeros((len(df), 0))
    S = ~np.isnan(X)
    Z, V = [], []
    for j in range(X.shape[1]):
        x, s = X[:, j], S[:, j]
        xs = x[s]
        cuts = {0.0} if ((xs > 0).any() and (xs <= 0).any()) else set()
        cuts |= {float(np.median(xs)), float(np.quantile(xs, 1 / 3)), float(np.quantile(xs, 2 / 3))}
        for t in sorted(cuts):
            with np.errstate(invalid="ignore"):
                hi = s & (x > t)
            Z += [hi, s & ~hi]
            V += [s, s]
    med = [float(np.median(X[S[:, j], j])) for j in range(X.shape[1])]
    for i in range(X.shape[1] if two_d else 0):
        with np.errstate(invalid="ignore"):
            hi_i = S[:, i] & (X[:, i] > med[i])
        for j in range(i + 1, X.shape[1]):
            with np.errstate(invalid="ignore"):
                hi_j = S[:, j] & (X[:, j] > med[j])
            both = S[:, i] & S[:, j]
            for a in (hi_i, both & ~hi_i):
                for b in (hi_j, both & ~hi_j):
                    Z.append(both & a & b)
                    V.append(both)
    if not Z:
        return np.zeros((len(df), 0), bool), np.zeros((len(df), 0), bool)
    return np.column_stack(Z), np.column_stack(V)


def _deltas(p, Zf, Rf, nz, nr, ok, sd=1.0):
    """t = (mean(rest) − mean(zone)) ÷ (sd · √(1/nz + 1/nr)) per column for result vector(s) p (n or R × n); sd = the cohort's sd
    (a shuffle moves results, never changes the set → sd is fixed). PURE."""
    with np.errstate(invalid="ignore", divide="ignore"):
        d = ((p @ Rf) / nr - (p @ Zf) / nz) / (sd * np.sqrt(1.0 / nz + 1.0 / nr))
    return np.where(ok, d, -np.inf)


def day_block_perms(days, r=R, seed=SEED):
    """→ list of r index arrays: the UTC days' fill blocks in a random order, laid back onto the fills in time order. PURE."""
    days = np.asarray(days)
    order = np.argsort(days, kind="stable")
    uniq = list(dict.fromkeys(days[order]))
    blocks = [order[days[order] == d] for d in uniq]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(r):
        src = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        perm = np.empty(len(days), dtype=int)
        perm[order] = src          # position k (in time order) takes the result of src[k]
        out.append(perm)
    return out


def block_line(df, zone, r=R, seed=SEED):
    """df = built-on fills (pct, day, entry_*), zone = the line's own mask → dict(n, nz, delta, p_single, p_search, verdict, k). PURE."""
    df = df.reset_index(drop=True)
    p = pd.to_numeric(df.pct, errors="coerce").to_numpy(float)
    keep = np.isfinite(p)
    df, p, zone = df[keep].reset_index(drop=True), p[keep], np.asarray(zone, bool)[keep]
    n, nz = len(df), int(zone.sum())
    base = dict(n=n, nz=nz, delta=float("nan"), p_single=float("nan"), p_1d=float("nan"), p_search=float("nan"), k=0)
    if n < MIN_N or nz < MIN_SIDE or n - nz < MIN_SIDE:
        return dict(base, verdict=FEW)
    cols = candidate_cols(df)
    k1 = family(df, cols, two_d=False)[0].shape[1]          # the 1D splits come first in the family
    Z, V = family(df, cols)
    Z = np.column_stack([zone, Z]) if Z.shape[1] else zone[:, None]
    V = np.column_stack([np.ones(n, bool), V]) if V.shape[1] else np.ones((n, 1), bool)
    Rm = V & ~Z
    nzk, nrk = Z.sum(0), Rm.sum(0)
    ok = (nzk >= MIN_SIDE) & (nrk >= MIN_SIDE)
    Zf, Rf = Z.astype(float), Rm.astype(float)
    sd = float(np.std(p, ddof=1)) or 1.0
    delta = float(p[~zone].mean() - p[zone].mean())
    obs = float(_deltas(p, Zf[:, :1], Rf[:, :1], nzk[:1], nrk[:1], ok[:1], sd)[0])
    P = np.stack([p[q] for q in day_block_perms(df.day.astype(str).to_numpy(), r, seed)])
    D = _deltas(P, Zf, Rf, nzk, nrk, ok, sd)
    single = D[:, 0]
    best = D.max(axis=1)
    best1 = D[:, :1 + k1].max(axis=1)
    return dict(base, delta=delta, t=obs, p_single=(1 + int((single >= obs).sum())) / (1 + r), p_1d=(1 + int((best1 >= obs).sum())) / (1 + r),
                p_search=(1 + int((best >= obs).sum())) / (1 + r), k=int(ok.sum()), verdict=verdict((1 + int((best >= obs).sum())) / (1 + r)))


def gain_line(d, days, chosen, r=R, seed=SEED):
    """d = DataFrame of per-trade Δ (one column per alternative tried), chosen = the line's column → dict as block_line (zone = n). PURE."""
    d = d.apply(pd.to_numeric, errors="coerce")
    ok = d.notna().all(axis=1).to_numpy()
    dropped = int((~ok).sum())          # a fill with ANY unpriced alternative leaves the test (shown on the cell, never silent)
    d, days = d[ok].reset_index(drop=True), np.asarray(days)[ok]
    n = len(d)
    base = dict(n=n, nz=n, delta=float("nan"), p_single=float("nan"), p_1d=float("nan"), p_search=float("nan"), k=d.shape[1], dropped=dropped)
    if n < MIN_N:
        return dict(base, verdict=FEW)
    M = d.to_numpy(float)
    obs = float(M[:, list(d.columns).index(chosen)].mean())
    uniq = list(dict.fromkeys(days))
    di = np.array([uniq.index(x) for x in days])
    rng = np.random.default_rng(seed)
    S = rng.choice((-1.0, 1.0), size=(r, len(uniq)))[:, di]          # each day's Δs flip together
    means = np.einsum("rn,nk->rk", S, M) / n
    ci = list(d.columns).index(chosen)
    ps = (1 + int((means[:, ci] >= obs).sum())) / (1 + r)
    pb = (1 + int((means.max(axis=1) >= obs).sum())) / (1 + r)
    return dict(base, delta=obs, p_single=ps, p_1d=pb, p_search=pb, verdict=verdict(pb))
