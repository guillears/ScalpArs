"""Best-case daily compound return per active sleeve + total, every current watch line switched on (in-sample upper bound)."""
import sys, os, time, itertools
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); SC = os.path.join(ROOT, "reports", "best_case")
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import review_math as RM, scout_fade_late_weak as LW, scout_ml_mkt_hot_watch as MH, watch_zones as WZ
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import master_exits as MX
import one_size as OS   # 📏 (DECISION_LOG 503) the one-size current-batch DCR
EXP = sys.argv[1]; NOW = int(time.time() * 1000)
B0, FLOWS = pd.Timestamp("2026-10-23 18:30"), [("2026-10-26T13:50:00", 7100.0)]
num = lambda d, c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
def sleeve(d): return d.entry_strategy.astype(str).where(d.entry_strategy.astype(str) != "MOMENTUM", "MOMENTUM_" + d.direction.astype(str))
# ── data
A = pd.read_csv(EXP, low_memory=False)
A = A[(A.status == "CLOSED") & (pd.to_datetime(A.opened_at, format="mixed") >= B0)].copy()
A["o"] = pd.to_datetime(A.opened_at, format="mixed", utc=True).dt.tz_convert(None)
A["pct"] = num(A, "pnl_percentage"); A["u"] = num(A, "pnl"); A["sl"] = sleeve(A); A["src"] = "B25"; A["era"] = "B25"
A["kk"] = A.opened_at.astype(str).str.replace(" ", "T").str[:19] + A.pair
_A_RAW = A[["closed_at", "pnl"]].copy()   # 📏 (503 deep review) the REAL account path: raw closes + pnl, before the exit re-pricing below
A, _b25_moved = MX.reprice_b25_exits(A, os.path.join(SC, "fb_combo.csv"))   # LITE C +2 closes → +3 · Flatbase → +5 / −10 (exits retired mid-batch)
M = pd.read_csv(os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv"), low_memory=False)
span = M.assign(o=pd.to_datetime(M.opened_at, format="mixed", utc=True).dt.tz_convert(None),
                c=pd.to_datetime(M.closed_at, format="mixed", utc=True).dt.tz_convert(None)).groupby("era").agg(a=("o", "min"), b=("c", "max"))
M = M[(M.stack_keep == True) & (M.is_probe != True)].copy()
M["o"] = pd.to_datetime(M.opened_at, format="mixed", utc=True).dt.tz_convert(None)
M["pct"] = num(M, "stack_pct"); M["sl"] = sleeve(M); M["src"] = "MASTER"
M, _n_long4 = MX.reprice_long_stop4(M)   # FRENZY_LONG master on the live −4 stop (master stack keeps the traded −3)
SLEEVES = sorted(A.sl.unique())
M = M[M.sl.isin(SLEEVES)]
# ── zone inputs
k1d, _ = LW.load_btc_1d(NOW, fetch=True)
_hot_sl = ["MOMENTUM_LONG", "FRENZY_WILLY", "FRENZY_WILLY_HT"]   # 🌡 (DL 489) Market-hot = V2 rolling top-100; a fill's own stamp wins, only the others are rebuilt
need = pd.concat([d[d.sl.isin(_hot_sl) & MH.needs_rebuild(d)].o for d in (A, M)])
for _try in range(6):   # the earlier pipeline steps can leave the shared per-minute Binance weight high: wait it out, never read a partial set
    _now = int(time.time() * 1000)   # fresh each try: a fixed NOW never lets a cooldown expire (deep review DL 448)
    if not len(need):
        closes, complete = pd.DataFrame(), True
        break
    closes, complete = MH.fetch_closes(int(need.min().value // 10**6), _now, days=sorted(set(need.dt.normalize())))
    if complete:
        break
    _cool = MH.LS._cooling(_now + 5 * 60_000)   # a 429 / 418 ban longer than 5 min: stop now, never wait it out here
    if _cool:
        sys.exit("Binance cooldown active (429 / 418) — re-run once it has expired")
    print(f"Market-hot read incomplete (try {_try + 1} / 6) — waiting 65 s for the Binance per-minute weight", file=sys.stderr)
    time.sleep(65)
if not complete:   # a partial coin set scores market-hot zones differently run to run — never flatter / move a line on a broken read
    sys.exit("Market-hot read incomplete (Binance error or rate-limit cooldown) — re-run in a few minutes")
br = MH.breadth_series(closes) if len(closes) else pd.Series(dtype=float)
def add_flags(d):
    d = d.copy()
    d["hot"] = MH.score(d[["o"] + ([MH.STAMP] if MH.STAMP in d else [])], br).zone.values
    d["late"] = LW.late_gap(k1d, d.opened_at.astype(str).str.replace(" ", "T").str[:19])
    # ML stop cooldown: a momentum long opened < 30 min after a momentum-long stop closed (same batch)
    d["cool"] = False
    ml = d[d.sl == "MOMENTUM_LONG"]
    for era, g in ml.groupby(d.get("era", pd.Series("x", index=d.index))):
        st = g[g.close_reason.astype(str).str.upper().str.contains("STOP")]
        sc = pd.to_datetime(st.closed_at, format="mixed", utc=True).dt.tz_convert(None).values
        for i, o in zip(g.index, g.o.values):
            d.loc[i, "cool"] = bool(((o > sc) & (o - sc < np.timedelta64(30, "m"))).any())
    return d
A, M = add_flags(A), add_flags(M)
def L(d):   # every current block line → boolean zone (unscored = False, never blocked) — one definition: scripts/watch_zones.py (DL 471)
    return WZ.zones(d, hot=d.hot, late=d.late, cool=d.cool)
LA, LM = L(A), L(M)
# per-fill zone flags for confidence.py (every line, both pools; unscored → False, as in the best case)
_z = []
for _src, _d, _LL in (("B25", A, LA), ("MASTER", M, LM)):
    for _sl, _lines in _LL.items():
        _g = _d[_d.sl == _sl]
        if len(_g):
            _f = pd.DataFrame({"src": _src, "sl": _sl, "o": _g.o, "pair": _g.pair, "pct": _g.pct})
            for _ln, _m in _lines.items():
                _f[_ln] = pd.Series(_m, index=_d.index).reindex(_g.index).fillna(False).astype(bool)
            _z.append(_f)
pd.concat(_z, ignore_index=True).to_csv(os.path.join(SC, "zones.csv"), index=False)
# FB_CONFIRM per fill (fb_combo.csv: cf510 = confirm entry at +5/−10, 0 = skipped)
FB = pd.read_csv(os.path.join(SC, "fb_combo.csv")); FB["kk"] = FB.k + FB.pair
fbmap = dict(zip(FB.kk, FB.cf510))
FX = pd.read_csv(os.path.join(SC, "flip_exits.csv")); FX["kk"] = FX.k + FX.pair
EXITS = ("TP1.5", "TP2", "TP3")
_bad = FX[(FX.status != "PRICED") | FX[list(EXITS)].isna().any(axis=1)]
if len(_bad):   # an unpriced flip would count as $0 downstream (fillna / groupby) — stop, never flatter the TP exit
    sys.exit("unpriced flips in flip_exits.csv — re-run flip_exits.py:\n" + _bad[["src", "k", "pair", "status"]].to_string())
_bad = FB[FB.cf510.isna()]
if len(_bad):   # a PENDING / unpriced FB_CONFIRM would be booked as blocked ($0) — stop
    sys.exit("unpriced Flatbase fills in fb_combo.csv:\n" + _bad[["k", "pair", "cf_why"]].to_string())
fxb = {kk: r for kk, r in FX[FX.src == "B25"].set_index("kk")[list(EXITS)].iterrows()}
fxm = {kk: r for kk, r in FX[FX.src == "MASTER"].set_index("kk")[list(EXITS)].iterrows()}
M["kk"] = M.opened_at.astype(str).str.replace(" ", "T").str[:19] + M.pair
# ── B25: ONE size (📏 DL 503, the headline) + the as-traded account path beside it
t0 = A.o.min(); t1 = pd.to_datetime(A.closed_at, format="mixed").max()
def b25_dcr(mask_sleeve, after_pct):
    """📏 (DECISION_LOG 503) ONE SIZE — every fill at today's intended size X; the headline and the line-picking number."""
    f = pd.Series(OS.frac(A.pct, A.sl.map(X)), index=A.index)
    for i, v in after_pct.items():
        f[i] = 0.0 if v is None else float(v) * X[A.at[i, "sl"]] / 100.0
    return OS.dcr(A.closed_at, f.values, np.asarray(mask_sleeve), t0, t1)
def b25_dcr_real(mask_sleeve, after_pct):
    """AS TRADED — the REAL account path (each fill's raw $ and close ÷ that morning's equity, the 10-26 top-up; a line's blocked fills
    removed); shown beside, never picks a line (503 deep review: was today's exits on real sizes)."""
    acc = A.copy(); acc["ua"] = pd.to_numeric(_A_RAW.pnl, errors="coerce").fillna(0.0); acc["closed_at"] = _A_RAW.closed_at
    for i, v in after_pct.items():
        acc.loc[i, "ua"] = 0.0 if v is None else v / 100 * float(acc.loc[i, "notional_value"])
    return RM.dcr_account(acc, mask_sleeve, "ua", 2900, t0, t1, FLOWS)
def b25_after(sl, lines, fbconf=False, ex=None):
    out = {}
    ix = A.index[A.sl == sl]
    z = pd.Series(False, index=A.index)
    for ln in lines:
        z |= LA[sl][ln].fillna(False).astype(bool)
    for i in ix:
        if z[i]:
            out[i] = None
        elif fbconf:
            v = fbmap.get(A.loc[i, "kk"])
            out[i] = None if (v is None or v != v or v == 0) else v
        elif ex:
            out[i] = float(fxb[A.loc[i, "kk"]][ex])   # KeyError = an unpriced flip: never silently kept as booked
    return out
# ── master intended size per era
import json as _json
_WILLY_INV = float(_json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"].get("frenzy_willy_invest_mult", 1.0) or 1.0)   # (487) ½ margin
# 📏 (503 deep review) today's size = notional ÷ the account equity when the fill opened (the bot sizes on current equity), not a fixed
# 2,900 / 10,000 step (that understated sizes 8–37 %, differently per sleeve)
_EQ_OPEN = pd.Series(OS.equity_at_open(A.opened_at, _A_RAW.closed_at, pd.to_numeric(_A_RAW.pnl, errors="coerce"), 2900.0, FLOWS), index=A.index)
X = {}
for sl in SLEEVES:
    g = A[A.sl == sl]   # 📏 (503) one definition of today's size, shared with scout_frenzy_duration.py
    X[sl] = OS.sleeve_x(g.notional_value, _EQ_OPEN[g.index], sl, g.get("leverage"), g.get("cell_multiplier"), _WILLY_INV)
    if not (X[sl] == X[sl] and X[sl] > 0):   # 📏 (503 review) a NaN / 0 size would silently read the sleeve at 0.000 %/day
        sys.exit(f"one-size X for {sl} is {X[sl]} (notional_value missing?) — fix the export before reading the best case")
def m_dcr(sl_list, drop_idx=frozenset(), repl=None):
    vals = []
    for era, (a, b) in span.iterrows():
        g = M[(M.era == era) & M.sl.isin(sl_list)]
        if not len(g):
            continue
        days = pd.date_range(a.normalize(), b.normalize())
        g = g[~g.index.isin(drop_idx)].copy()
        if repl:
            hit = [i for i in g.index if i in repl]
            g.loc[hit, "pct"] = [repl[i] for i in hit]
        c = pd.to_datetime(g.closed_at, format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
        r = (g.pct * g.sl.map(X) / 100).groupby(c).sum().reindex(days).fillna(0)
        vals.append(((1 + r).prod() ** (1 / len(days)) - 1) * 100)
    return float(np.mean(vals)) if vals else float("nan"), len(vals)
def stat(g): return f"{len(g)} · {(g>0).mean():.0%} · {g.mean():+.2f}" if len(g) else "0"
rows = []
SCEN = {"A": ({}, set(), {}), "B": ({}, set(), {})}
# 🤝 (DECISION_LOG 501, operator 2026-11-08: "if we see master and current as a whole batch of continuous trading, RETRY_BTC1H seems better") "helps both" gives
# master a veto only when master has enough trades to mean something: a combo that changes < MASTER_MIN_CHANGED master fills is judged on
# master + current batch POOLED (Σ % per fill over the kept fills of both, today's exits) instead — 2 master fills must not outvote 19.
MASTER_MIN_CHANGED = 5
BASIS, NCHG = {}, {}
REAL, KSUM = {}, {}   # 📏 (503) per (scen, sleeve): as-traded DCR before / best · kept Σ % before / best / N
print(f"Market-hot read complete={complete} · X (notional/account) = " + ", ".join(f"{k} {v:.2f}" for k, v in X.items()))
for sl in SLEEVES:
    msk = A.sl == sl
    before = b25_dcr(msk, {})
    mb, nm = m_dcr([sl]) if (M.sl == sl).any() else (float("nan"), 0)
    lines = list(LA.get(sl, {}).keys()) + (["FB_CONFIRM"] if sl == "FLATBASE_SHORT" else []) + []
    print(f"\n### {sl}  B25 {stat(A[msk].pct)} · DCR {before:+.3f}   |   master {stat(M[M.sl == sl].pct)} · DCR {mb:+.3f} ({nm} batches)")
    res = []
    for r in range(1, len(lines) + 1):
        for combo in itertools.combinations(lines, r):
            exs = [c[5:] for c in combo if c.startswith("EXIT ")]
            if len(exs) > 1:
                continue
            ex = exs[0] if exs else None
            blk = [c for c in combo if c != "FB_CONFIRM" and not c.startswith("EXIT ")]
            aft = b25_after(sl, blk, "FB_CONFIRM" in combo, ex)
            mrep = {i: float(fxm[M.loc[i, "kk"]][ex]) for i in M.index[M.sl == sl]} if ex else None
            d = b25_dcr(msk, aft)
            if blk:
                z = pd.Series(False, index=M.index)
                for ln in blk:
                    z |= LM[sl][ln].fillna(False).astype(bool)
                dm, _ = m_dcr([sl], frozenset(M.index[z & (M.sl == sl)]), mrep)
            else:
                dm = m_dcr([sl], frozenset(), mrep)[0] if mrep else mb
            kept = [v for i, v in aft.items() if v is not None] + [A.loc[i, "pct"] for i in A.index[msk] if i not in aft]
            _mg = M.index[M.sl == sl]
            _mdrop = set(M.index[z & (M.sl == sl)]) if blk else set()
            _mv = {i: (float(mrep[i]) if mrep else float(M.at[i, "pct"])) for i in _mg if i not in _mdrop}
            _nch = len(_mdrop) + sum(1 for i, v in _mv.items() if not (abs(v - float(M.at[i, "pct"])) <= 1e-9))   # NaN → value counts as changed (review)
            res.append((d, combo, dm, len(kept), _nch, float(np.nansum(kept)) + float(np.nansum(list(_mv.values())))))
            if r == 1:
                print(f"   {combo[0]:<26} B25 DCR {before:+.3f} → {d:+.3f} · kept {len(kept)}/{msk.sum()}   master {mb:+.3f} → {dm:+.3f}")
    _pool0 = float(np.nansum(A[msk].pct)) + float(np.nansum(M[M.sl == sl].pct))
    res.append((before, (), mb, int(msk.sum()), 0, _pool0))   # "keep as is" is always an option
    _mok = lambda x: x[2] != x[2] or x[2] >= mb - 1e-9
    _pok = lambda x: x[4] < MASTER_MIN_CHANGED and x[5] >= _pool0 - 1e-9
    if len(lines) == 0:
        print("   (no current watch line)")
    for tag, pool in (("A", res), ("B", [x for x in res if (_mok(x) or _pok(x)) and x[0] >= before - 1e-9])):
        d, combo, dm, nk, nch, pl = max(pool, key=lambda x: x[0])
        BASIS[(tag, sl)] = "" if tag == "A" or not combo or _mok((d, combo, dm)) else "pooled"
        NCHG[(tag, sl)] = nch
        _why = f" (pooled: master changes only {nch} fill(s) < {MASTER_MIN_CHANGED} → master + current batch Σ {_pool0:+.2f} → {pl:+.2f} %)" if BASIS[(tag, sl)] else ""
        print(f"   ★ {tag} {'best on B25' if tag == 'A' else 'helps both'}: {' + '.join(combo) or 'no line'} → B25 {before:+.3f} → {d:+.3f} · master {mb:+.3f} → {dm:+.3f}{_why}")
        rows.append((tag, sl, before, d, mb, dm, " + ".join(combo)))
        exs = [c[5:] for c in combo if c.startswith("EXIT ")]; ex = exs[0] if exs else None
        blk = [c for c in combo if c != "FB_CONFIRM" and not c.startswith("EXIT ")]
        _aft = b25_after(sl, blk, "FB_CONFIRM" in combo, ex)
        SCEN[tag][0].update(_aft)
        REAL[(tag, sl)] = (b25_dcr_real(msk, {}), b25_dcr_real(msk, _aft))
        _k = [v for i, v in _aft.items() if v is not None] + [A.loc[i, "pct"] for i in A.index[msk] if i not in _aft]
        _k = [v for v in _k if v == v]   # (503 review) an unpriced (NaN) fill counts neither in N nor in Σ
        KSUM[(tag, sl)] = (float(np.nansum(A[msk].pct)), float(np.nansum(_k)), len(_k))
        if ex:
            SCEN[tag][2].update({i: float(fxm[M.loc[i, "kk"]][ex]) for i in M.index[M.sl == sl]})
        z = pd.Series(False, index=M.index)
        for ln in blk:
            z |= LM[sl][ln].fillna(False).astype(bool)
        SCEN[tag][1].update(set(M.index[z & (M.sl == sl)]))
allm = pd.Series(True, index=A.index)
tb = b25_dcr(allm, {}); mt_b, n = m_dcr(SLEEVES)
for tag in ("A", "B"):
    ta = b25_dcr(allm, SCEN[tag][0]); mt_a, _ = m_dcr(SLEEVES, frozenset(SCEN[tag][1]), SCEN[tag][2])
    print(f"\nTOTAL {tag}: B25 one size {tb:+.3f} → {ta:+.3f} %/day · master intended size {mt_b:+.3f} → {mt_a:+.3f} %/day ({n} batches)")
# without WILLY's B25 (the biggest single driver): does the rest still move?
nw = A.sl != "FRENZY_WILLY"
print(f"TOTAL B ex-WILLY B25: {b25_dcr(nw, {}):+.3f} → {b25_dcr(nw, {i: v for i, v in SCEN['B'][0].items() if A.loc[i, 'sl'] != 'FRENZY_WILLY'}):+.3f}")
pd.DataFrame(rows, columns=["scen", "sleeve", "b25_before", "b25_best", "m_before", "m_best", "lines"]).assign(
    basis=[BASIS.get((r[0], r[1]), "") for r in rows], m_changed=[NCHG.get((r[0], r[1]), 0) for r in rows],
    b25_before_real=[REAL[(r[0], r[1])][0] for r in rows], b25_best_real=[REAL[(r[0], r[1])][1] for r in rows],
    b25_sum_before=[KSUM[(r[0], r[1])][0] for r in rows], b25_sum_best=[KSUM[(r[0], r[1])][1] for r in rows],
    b25_n_best=[KSUM[(r[0], r[1])][2] for r in rows]).to_csv(os.path.join(SC, "best_dcr_noflip.csv"), index=False)

# ── re-sizing on top of scenario B — the DL 502 5 % cap (📏 DL 503): RETRY / Flatbase up, LITE down; WILLY / FRENZY_LONG never sized up
def b25_sized(after, size):
    """📏 (DECISION_LOG 503) the re-sizes at ONE size: account fractions (% × X ÷ 100) × the size factor (WILLY: target × ÷ today's 5×)."""
    acc = A.copy(); acc["ua"] = OS.frac(acc.pct, acc.sl.map(X))
    for i, v in after.items():
        acc.loc[i, "ua"] = 0.0 if v is None else float(v) * X[acc.at[i, "sl"]] / 100.0
    for sl, f in size.items():
        m = acc.sl == sl
        acc.loc[m, "ua"] = acc.loc[m, "ua"] * (f / 5.0 if sl == "FRENZY_WILLY" else f)   # X WILLY is at today's 5× (kept for a WILLY row; none today, DL 503) (as m_sized)
    return acc
def acc_dcr(acc, mask):
    return OS.dcr(acc.closed_at, acc.ua.values, np.asarray(mask), t0, t1)
def m_sized(drop, repl, size):
    global X
    X0 = dict(X)
    for sl, f in size.items():
        X[sl] = X0[sl] * (f / 5.0 if sl == "FRENZY_WILLY" else f)   # master WILLY X is at 5×
    try:
        return m_dcr(SLEEVES, frozenset(drop), repl)[0]
    finally:
        X = X0
B = SCEN["B"]
# 📏 (DECISION_LOG 503, operator 11-08: "maybe we should change all this, for the recently talked items") the re-size rows = the DL 502
# 5 % cap: each sleeve sized so ONE stop costs 5 % of the account at today's size (factor = 5 ÷ (X × stop %)); only sleeves above their
# breakeven WR step up (RETRY, Flatbase); LITE steps DOWN (its stop costs > 5 %). WILLY / FRENZY_LONG are below breakeven → never sized up.
_TH = _json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
_STOP = {"FRENZY_LITE_RETRY": float(_TH.get("frenzy_stop_pct", 3.0) or 3.0), "FRENZY_LITE": float(_TH.get("frenzy_stop_pct", 3.0) or 3.0),
         "FLATBASE_SHORT": float(_TH.get("flatbase_stop_pct", 10.0) or 10.0)}
CAP5 = {sl: round(OS.cap_factor(X[sl], st), 2) for sl, st in _STOP.items() if sl in X and X[sl] > 0}
print("RESIZE FACTORS (5 % cap, one stop = 5 % of the account): " + " · ".join(f"{sl} ×{f:.2f} (stop now {X[sl] * _STOP[sl]:.1f} %)" for sl, f in CAP5.items()))
# 📏 (DL 504, operator 11-09: "re check the other sleeves (non frenzy ones)" → "in ② only add Momentum Long") Momentum Long ×5 / ×10 —
# Kelly- and cluster-limited, not cap-limited (one stop ≈ 0.17 % of the account; up to 4 opened within 10 min). Spike Fade (×1.4 at its cap),
# Momentum Short and FAN flips are size WATCH lines only (CURRENT_STATE), not rows here.
_ML_STOP = 0.70   # Momentum Long's median booked STOP loss % (DL 504, reports/SIZING_5PCT_CAP_2026-11-08.md)
print(f"RESIZE FACTORS 2 (DL 504): MOMENTUM_LONG ×5 / ×10 (one stop now ≈ {X.get('MOMENTUM_LONG', 0) * _ML_STOP:.2f} % of the account · ×5 ≈ {5 * X.get('MOMENTUM_LONG', 0) * _ML_STOP:.2f} % · ×10 ≈ {10 * X.get('MOMENTUM_LONG', 0) * _ML_STOP:.2f} %)")
_C3 = {**CAP5, **({"MOMENTUM_LONG": 5.0} if "MOMENTUM_LONG" in X else {})}
RESIZES = (("C1 5 % cap: RETRY + Flatbase up, LITE down", dict(CAP5)),
           ("C2 5 % cap: RETRY + Flatbase up only", {k: v for k, v in CAP5.items() if v > 1}),
           ("C3 C1 + Momentum Long ×5", _C3),
           ("C4 C3 with Momentum Long ×10", {**_C3, **({"MOMENTUM_LONG": 10.0} if "MOMENTUM_LONG" in X else {})}))
for nm, size in RESIZES:
    acc = b25_sized(B[0], size)
    per = {sl: acc_dcr(acc, acc.sl == sl) for sl in size}
    tot = acc_dcr(acc, pd.Series(True, index=acc.index))
    print(f"{nm}: B25 total {tot:+.3f} %/day · " + " · ".join(f"{k} {v:+.3f}" for k, v in per.items()) + f" | master total {m_sized(B[1], B[2], size):+.3f}")
# Spike Fade size line: HEALTHY_BEAR fades at 1× (traded 2×) — alone and on top of OBS-F1
hbA = A.sl.eq("SPIKE_FADE") & A.entry_btc_regime.astype(str).eq("HEALTHY_BEAR")
hbM = M.sl.eq("SPIKE_FADE") & M.entry_btc_regime.astype(str).eq("HEALTHY_BEAR")
fm = A.sl == "SPIKE_FADE"
def fade_case(block):
    acc = A.copy(); acc["ua"] = OS.frac(acc.pct, acc.sl.map(X))   # 📏 (503) one size
    acc.loc[hbA, "ua"] = acc.loc[hbA, "ua"] * 0.5
    if block is not None:
        acc.loc[block & fm, "ua"] = 0.0
    return acc_dcr(acc, fm)
def fade_m(block, half):
    repl = {i: M.loc[i, "pct"] * 0.5 for i in M.index[hbM]} if half else None
    drop = frozenset(M.index[block & M.sl.eq("SPIKE_FADE")]) if block is not None else frozenset()
    return m_dcr(["SPIKE_FADE"], drop, repl)[0]
f1A = LA["SPIKE_FADE"]["OBS-F1 late weak spike"].fillna(False).astype(bool); f1M = LM["SPIKE_FADE"]["OBS-F1 late weak spike"].fillna(False).astype(bool)
g = A[hbA]; gm = M[hbM]
print(f"\nFADE HB: B25 HB {len(g)} · {(g.pct>0).mean():.0%} · {g.pct.mean():+.3f} · ${g.u.sum():+.0f} | master HB {len(gm)} · {(gm.pct>0).mean():.0%} · {gm.pct.mean():+.3f}")
print(f"  HB 1× alone: B25 {b25_dcr(fm, {}):+.3f} → {fade_case(None):+.3f} · master {m_dcr(['SPIKE_FADE'])[0]:+.3f} → {fade_m(None, True):+.3f}")
print(f"  OBS-F1 alone: B25 → {b25_dcr(fm, b25_after('SPIKE_FADE', ['OBS-F1 late weak spike'])):+.3f} · master → {fade_m(f1M, False):+.3f}")
print(f"  OBS-F1 + HB 1×: B25 → {fade_case(f1A):+.3f} · master → {fade_m(f1M, True):+.3f}")
for r in g.sort_values("o").itertuples():
    print(f"    {r.o:%m-%d %H:%M} {r.pair} {r.pct:+.2f} % · ${r.u:+.0f} → 1× ${r.u * 0.5:+.0f} · OBS-F1 zone {bool(f1A[r.Index])}")
