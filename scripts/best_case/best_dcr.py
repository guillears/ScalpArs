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
_bad = FX[(FX.status != "PRICED") | FX[list(EXITS)].isna().any(axis=1)
         | (FX[[f"h_{x}" for x in EXITS]].isna().any(axis=1) if f"h_{EXITS[0]}" in FX else False)]   # a missing hold stops too (480 review)
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
    lines = list(LA.get(sl, {}).keys()) + (["FB_CONFIRM"] if sl == "FLATBASE_SHORT" else []) + ([f"EXIT {x}" for x in EXITS] if sl.startswith("FLIP") else [])
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
# ── hold time + trades per day per sleeve, before → best (operator 11-06: "add avg duration before and after per sleeve and also avg trades
# per day"), scenario B. Before = every fill on today's exits. Best = the fills the best line keeps; an exit line holds to its own exit
# (flip_exits.csv h_<exit>), FB_CONFIRM from its confirmed entry to its exit (fb_combo.csv h_cf510 − wait_min). Trades per day: current
# batch = N ÷ days since the batch start (to the last open/close in the export); master = N ÷ the calendar days of the batches the sleeve
# traded in (its own live days, never the whole master span). MASTER HOLD = AS TRADED: re-priced master exits (stack_pct ≠ booked: most
# WILLY, FRENZY_LONG −4, some LITE / fades) keep the traded close time, as do current-batch LITE C re-walks (the walks carry no exit time).
def _hold(d):
    return (pd.to_datetime(d.closed_at, format="mixed", utc=True).dt.tz_convert(None) - d.o).dt.total_seconds() / 60
_raw = pd.read_csv(EXP, low_memory=False, usecols=["opened_at", "closed_at"])   # the export's own times (re-priced Flatbase walks move closed_at)
_end = max(pd.to_datetime(_raw.opened_at, format="mixed", utc=True).dt.tz_convert(None).max(),
           pd.to_datetime(_raw.closed_at, format="mixed", utc=True).dt.tz_convert(None).max())
B25_DAYS = (_end - B0).total_seconds() / 86400
FXH = {(s_, kk): r for (s_, kk), r in FX.set_index(["src", "kk"])[[f"h_{x}" for x in EXITS]].iterrows()} if "h_TP2" in FX else None
FBH = dict(zip(FB.kk, pd.to_numeric(FB.get("h_cf510"), errors="coerce") - pd.to_numeric(FB.get("wait_min"), errors="coerce").fillna(0)))
HOLD = []
for tag, sl, *_r, combo in [r for r in rows if r[0] == "B"]:
    ln = combo.split(" + ") if combo else []
    ex = next((c[5:] for c in ln if c.startswith("EXIT ")), None)
    blk = [c for c in ln if c != "FB_CONFIRM" and not c.startswith("EXIT ")]
    if ex and FXH is None:
        sys.exit("flip_exits.csv has no hold columns (h_TP*) — re-run flip_exits.py")
    g = A[A.sl == sl]; hb = _hold(g)
    aft = b25_after(sl, blk, "FB_CONFIRM" in ln, ex)
    kept = [i for i in g.index if not (i in aft and aft[i] is None)]
    if ex:
        ha = pd.Series([FXH[("B25", A.at[i, "kk"])][f"h_{ex}"] for i in kept], index=kept, dtype=float)
    elif "FB_CONFIRM" in ln:
        ha = pd.Series([FBH.get(A.at[i, "kk"], np.nan) for i in kept], index=kept, dtype=float)
        if ha.isna().any():   # a kept Flatbase fill without a confirm hold would silently drop out of the mean (480 review)
            sys.exit("kept FB_CONFIRM fills without a hold in fb_combo.csv (h_cf510) — re-run fb_combo.py")
    else:
        ha = hb.reindex(kept)
    gm = M[M.sl == sl]
    z = pd.Series(False, index=M.index)
    for b_ in blk:
        z |= LM[sl][b_].fillna(False).astype(bool)
    keptm = gm.index[~z.reindex(gm.index).fillna(False)]
    hmb = _hold(gm)
    hma = (pd.Series([FXH[("MASTER", M.at[i, "kk"])][f"h_{ex}"] for i in keptm], index=keptm, dtype=float) if ex else hmb.reindex(keptm))
    mdays = sum(len(pd.date_range(a.normalize(), b.normalize())) for era, (a, b) in span.iterrows() if (gm.era == era).any())
    HOLD.append(dict(sleeve=sl, b25_n0=len(g), b25_n1=len(kept), b25_h0=hb.mean(), b25_h1=ha.mean(), b25_days=B25_DAYS,
                     m_n0=len(gm), m_n1=len(keptm), m_h0=hmb.mean(), m_h1=hma.mean(), m_days=mdays))
pd.DataFrame(HOLD).to_csv(os.path.join(SC, "best_hold.csv"), index=False)
# ── ①c ⏱ TIME CAP ON THE BEST LINE (DECISION_LOG 497, operator 11-07: "study this in best case, like if it helps to winners on the filters")
# Per FRENZY sleeve, per cap X below its live cap (frenzy_caps.py: REAL path for current-batch fills on today's exit, a paired walk for the
# rest): does the shorter cap still pay on the fills the sleeve's best line KEEPS (scenario B), or only on fills a filter already blocks?
# ALL = the cap on every fill · KEPT = best line before → best line + cap · cut = the kept fills the cap changes, winners vs losers (base > 0).
# Shown only — never switched into the scenario totals. "helps both" = better on the current batch's kept fills AND never worse on master's,
# with ≥ 1 master fill changed; a block whose walk parity failed (< frenzy_caps.PARITY_MIN) is "not read" and never "helps both". P = day-block
# bootstrap on the kept fills (master only when read), by close day. 🎲 luck: the CURRENT-BATCH kept fills' Δ for every X tried vs the same
# search on day-shuffled results (luck_test.gain_line) — the frame the cap is chosen on; only a cap that gains is a candidate.
import scout_frenzy_duration as FD
import luck_test as LT0
import frenzy_caps as FCm
if pd.Timestamp(FCm.B0) != B0:   # one batch start for both steps — a drift would silently mix cohorts
    sys.exit(f"best_dcr B0 {B0} ≠ frenzy_caps B0 {FCm.B0} (sweep_separators.BATCH_START) — update both for the new batch")
FC = pd.read_csv(os.path.join(SC, "frenzy_caps.csv"))
FC["kk"] = FC.k.astype(str) + FC.pair.astype(str)
if FC.status.astype(str).str.startswith("UNPRICED").any():   # an unpriced fill would count as booked — stop, never flatter a cap
    sys.exit("unpriced FRENZY fills in frenzy_caps.csv — re-run frenzy_caps.py --fetch:\n"
             + FC[FC.status.astype(str).str.startswith("UNPRICED")][["src", "sl", "k", "pair", "status"]].to_string())
if FC.duplicated(["src", "kk"]).any():
    sys.exit("duplicate fills in frenzy_caps.csv (src, opened_at, pair) — the cap join would be ambiguous")
CMAP = {(s_, kk): r for (s_, kk), r in FC.set_index(["src", "kk"]).iterrows()}
PAR = FC[FC.method == "WALK"].groupby(["src", "sl"]).apply(lambda g: float((g.status != "PARITY MISS (kept as booked)").mean())).to_dict()
_CAPS = FD.caps()
CROWS = []
def _val(src, kk, col):
    r = CMAP.get((src, kk))
    if r is None:
        raise KeyError(f"{src} {kk} missing from frenzy_caps.csv — re-run frenzy_caps.py")
    v = r.get(col)
    if v is None or v != v:
        raise KeyError(f"{src} {kk} {col} unpriced in frenzy_caps.csv")
    return float(v)
def _vblock(sl, cols, extra, SINK, what):
    """one FRENZY sleeve, every exit variant column in cols (frenzy_caps.csv) on top of its best line → rows into SINK (①c caps · ①d late TP).
    ALL / KEPT daily compound, winners vs losers cut, P, helps both, 🎲 luck over every variant tried on the current batch's kept trades."""
    combo = next((r[6] for r in rows if r[0] == "B" and r[1] == sl), "")
    blk = [c for c in (combo.split(" + ") if combo else []) if c != "FB_CONFIRM" and not c.startswith("EXIT ")]
    msk = A.sl == sl
    ia = list(A.index[msk]); im = list(M.index[M.sl == sl])
    base_after = b25_after(sl, blk)                                   # blocked → None (scenario B's best line)
    keep_a = [i for i in ia if not (i in base_after and base_after[i] is None)]
    zm = pd.Series(False, index=M.index)
    for ln in blk:
        zm |= LM[sl][ln].fillna(False).astype(bool)
    drop_m = frozenset(M.index[zm & (M.sl == sl)])
    keep_m = [i for i in im if i not in drop_m]
    m_ok = PAR.get(("MASTER", sl), 1.0) >= FCm.PARITY_MIN if im else False
    b_ok = PAR.get(("B25", sl), 1.0) >= FCm.PARITY_MIN             # the current batch's walked (re-priced) fills too
    b0 = b25_dcr(msk, {}); bk0 = b25_dcr(msk, base_after)
    m0 = m_dcr([sl])[0] if im else float("nan"); mk0 = m_dcr([sl], drop_m)[0] if im else float("nan")
    DLT = {}
    mine = []
    for col in cols:
        ca = {i: _val("B25", A.at[i, "kk"], col) for i in ia}
        cm = {i: _val("MASTER", M.at[i, "kk"], col) for i in im}
        d_all = b25_dcr(msk, ca)
        d_kept = b25_dcr(msk, {**base_after, **{i: ca[i] for i in keep_a}})
        dm_all = m_dcr([sl], frozenset(), cm)[0] if im else float("nan")
        dm_kept = m_dcr([sl], drop_m, {i: cm[i] for i in keep_m})[0] if im else float("nan")
        da = pd.Series({i: ca[i] - float(A.at[i, "pct"]) for i in keep_a}, dtype=float)
        dmm = pd.Series({i: cm[i] - float(M.at[i, "pct"]) for i in keep_m}, dtype=float)
        def cut(dd, src):
            ch = dd[dd.abs() > 1e-9]
            bs = (A if src == "B25" else M).loc[ch.index, "pct"]
            w, l_ = ch[bs > 0], ch[bs <= 0]
            return len(w), float(w.sum()), len(l_), float(l_.sum())
        _cd = lambda d_, i: str(pd.to_datetime(d_.at[i, "closed_at"], format="mixed", utc=True).date())   # close day (boot_p's unit)
        pool = pd.concat([da, dmm.rename(lambda i: f"m{i}")]) if m_ok else da   # a master read that failed parity never enters P
        days = [_cd(A, i) for i in da.index] + ([_cd(M, i) for i in dmm.index] if m_ok else [])
        DLT[col] = da                                                         # 🎲 the luck test runs on the frame the variant is CHOSEN on
        pb = FD.boot_p(pool.values, days) if (pool.abs() > 1e-9).any() else float("nan")
        m_changed = int((dmm.abs() > 1e-9).sum())
        both = (m_ok and b_ok and m_changed > 0 and d_kept > bk0 + 1e-9 and dm_kept >= mk0 - 1e-9)   # master must be read AND touched
        r_ = dict(sleeve=sl, line=combo, **extra(col), b25_all0=b0, b25_all1=d_all, b25_kept0=bk0, b25_kept1=d_kept,
                  m_all0=m0, m_all1=dm_all if m_ok else float("nan"), m_kept0=mk0, m_kept1=dm_kept if m_ok else float("nan"),
                  m_read=("ok" if m_ok else ("no master trades" if not im else f"not read (walk parity {PAR.get(('MASTER', sl), 0):.0%})")),
                  b_read=("ok" if b_ok else f"not read (walk parity {PAR.get(('B25', sl), 0):.0%})"), m_changed=m_changed,
                  n_kept_b25=len(keep_a), n_kept_m=len(keep_m),
                  **dict(zip(("b25_w_cut", "b25_w_d", "b25_l_cut", "b25_l_d"), cut(da, "B25"))),
                  **dict(zip(("m_w_cut", "m_w_d", "m_l_cut", "m_l_d"), cut(dmm, "M"))),
                  p_boot=pb, helps_both=bool(both), luck="", col=col)
        mine.append(r_)
        print(f"{what} {sl} {col} (best line: {combo or 'none'}): B25 all {b0:+.3f} → {d_all:+.3f} · kept {bk0:+.3f} → {d_kept:+.3f} | "
              f"master all {m0:+.3f} → {dm_all:+.3f} · kept {mk0:+.3f} → {dm_kept:+.3f} ({'ok' if m_ok else 'not read'}) · P {pb:.2f} · helps both {both}")
    if cols and len(DLT):
        dd = pd.DataFrame({c: DLT[c] for c in cols})                          # current-batch kept trades (where the variant was chosen)
        days = [str(pd.to_datetime(A.at[i, "closed_at"], format="mixed", utc=True).date()) for i in dd.index]
        gain = [r for r in mine if r["b25_kept1"] > r["b25_kept0"] + 1e-9 and r["b_read"] == "ok"]   # only a variant that GAINS on the kept trades
        if not gain:
            for r in mine:
                r["luck"] = f"n/a — no {what.lower()} gains on the kept trades"
        else:
            best = max(gain, key=lambda r: r["b25_kept1"])
            lt = LT0.gain_line(dd, days, best["col"])
            for r in mine:
                r["luck"] = (f"{lt['verdict']} (chosen {best['label']} · p {lt['p_search']:.2f} · n {lt['n']}"
                             + ("" if what == "CAP" else f" · {len(cols)} tried") + ")"
                             if lt["verdict"] != LT0.FEW else f"too few ({lt['n']} trades)")
    SINK.extend(mine)
for sl in [x for x in SLEEVES if x in FD.SLEEVES]:
    xs = [x for x in FD.XS if x < _CAPS[sl]]
    _vblock(sl, [f"CAP{x}" for x in xs], lambda c, sl=sl: dict(cap=int(c[3:]), live_cap=_CAPS[sl], label=f"{int(c[3:])} min"), CROWS, "CAP")
pd.DataFrame(CROWS).to_csv(os.path.join(SC, "best_caps.csv"), index=False)
# ── ①d ⏬ LATE TP ON THE BEST LINE (operator 11-07: "for the sleeves with 12h max duration, what happens if after X time we lower the TP? …
# together with the best case filters … only for the winners in the best case filters") — frenzy_caps.py LT<X>_<L>: today's exit and cap,
# the resting TP lowered to L after X minutes. Same frame, P, helps both and 🎲 luck as ①c (luck = every X × L tried for the sleeve).
LATE_ROWS = []
for sl in [x for x in SLEEVES if x in FD.SLEEVES]:
    lv = FCm.late_variants(sl, _CAPS[sl])
    if lv:
        tp_ = FCm.exit_spec(sl)["tp"]
        ls = FC[FC.sl == sl]["late_status"].astype(str) if "late_status" in FC.columns else pd.Series(dtype=str)
        if ls.str.startswith("UNPRICED").any():   # a late-TP hole → this sleeve's ①d is not read (never priced as booked)
            print(f"LATE TP {sl}: not read — {int(ls.str.startswith('UNPRICED').sum())} fill(s) with 1m bars missing")
            LATE_ROWS.append(dict(sleeve=sl, line="", after=float("nan"), late_tp=float("nan"), tp=tp_, live_cap=_CAPS[sl],
                              b_read="not read (1m bars missing — re-run frenzy_caps.py --fetch)", m_read="–"))
            continue
        _vblock(sl, [FCm.late_col(x, l_) for x, l_ in lv],
                lambda c, tp_=tp_, sl=sl: dict(after=int(c[2:].split("_")[0]), late_tp=float(c.split("_")[1]), tp=tp_, live_cap=_CAPS[sl],
                                               label=f"{int(c[2:].split('_')[0])} min → +{float(c.split('_')[1]):g}"), LATE_ROWS, "LATE TP")
pd.DataFrame(LATE_ROWS).to_csv(os.path.join(SC, "best_late.csv"), index=False)
allm = pd.Series(True, index=A.index)
tb = b25_dcr(allm, {}); mt_b, n = m_dcr(SLEEVES)
print(f"AS TRADED TODAY: B25 {b25_dcr_real(allm, {}):+.3f} %/day (account path, each fill's actual size — shown, never compared)")
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
    b25_n_best=[KSUM[(r[0], r[1])][2] for r in rows]).to_csv(os.path.join(SC, "best_dcr.csv"), index=False)

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
# ── CLEAN-ONLY (operator 11-04: "build the clean-only column"): scenario B with ONLY the lines whose CLEAN fills pass the DECISION_LOG 447
# ladder (past checkpoint ∧ 95 % day range on the line's side → no haircut); a re-size counts only when its gating line passed (FRENZY_LONG's
# DL 317 stop read is not in this view → never on here). Grows toward the full number as lines pass; equals today while none has.
import confidence as CF
RESIZE_GATE = {"FRENZY_LITE_RETRY": "RETRY_BTC1H", "FLATBASE_SHORT": "FB_CONFIRM", "MOMENTUM_LONG": "ML_MKT_HOT"}   # a step UP waits for its line · a step DOWN (risk cap) always counts
_best = [(r[1], r[6]) for r in rows if r[0] == "B" and r[6]]
CALLS = CF.line_calls([(sl, ln) for sl, combo in _best for ln in combo.split(" + ")])
CS = ({}, set(), {})
for sl, combo in _best:
    keep = [ln for ln in combo.split(" + ") if CALLS.get((sl, ln)) == "PASS"]
    exs = [c[5:] for c in keep if c.startswith("EXIT ")]; ex = exs[0] if exs else None
    blk = [c for c in keep if c != "FB_CONFIRM" and not c.startswith("EXIT ")]
    CS[0].update(b25_after(sl, blk, "FB_CONFIRM" in keep, ex))
    if ex:
        CS[2].update({i: float(fxm[M.loc[i, "kk"]][ex]) for i in M.index[M.sl == sl]})
    z = pd.Series(False, index=M.index)
    for ln in blk:
        z |= LM[sl][ln].fillna(False).astype(bool)
    CS[1].update(set(M.index[z & (M.sl == sl)]))
print("\nCLEAN CALLS: " + " · ".join(f"{sl}:{ln} {c}" for (sl, ln), c in CALLS.items()))
print(f"CLEAN TOTAL B: B25 one size {tb:+.3f} → {b25_dcr(allm, CS[0]):+.3f} %/day · master intended size {mt_b:+.3f} → "
      f"{m_dcr(SLEEVES, frozenset(CS[1]), CS[2])[0]:+.3f}")
for nm, size in RESIZES:
    sz = {sl: f for sl, f in size.items() if f < 1 or (RESIZE_GATE.get(sl) and CALLS.get((sl, RESIZE_GATE[sl])) == "PASS")}
    acc = b25_sized(CS[0], sz)
    print(f"CLEAN {nm.split()[0]}: B25 total {acc_dcr(acc, pd.Series(True, index=acc.index)):+.3f} %/day"
          f" | master total {m_sized(CS[1], CS[2], sz):+.3f} · re-sized: {', '.join(sz) or 'none'}")
# ── 🎲 LUCK TEST (DECISION_LOG 473, operator 11-05: "why not doing for every rule in watchlist?"): each best line's BUILT-ON trades
# (master + current batch before its freeze) against the same search run on day-shuffled results (luck_test.py). LUCK-CHECKED totals =
# scenario B with every line whose history is NOT ABOVE LUCK switched off (its history counts for nothing); the locked 30–50 % cut is
# unchanged for the others; a re-size is off when ANY of its sleeve's best lines is NOT ABOVE LUCK (deep review 473: FRENZY_LONG ×5 has
# no gating line of its own, so the sleeve's lines decide).
import luck_test as LT
LROWS = []
for sl, combo in _best:
    for ln in combo.split(" + "):
        fz = CF.freeze_ts(CF.LINES[ln if not ln.startswith("EXIT ") else "EXIT TP2"][0]) if (ln in CF.LINES or ln.startswith("EXIT ")) else None
        if ln == "FB_CONFIRM":
            g = FB.assign(o=pd.to_datetime(FB.k, format="mixed"))
            g = g if fz is None else g[g.o < fz]
            if "cf_why" in g:   # confirm-entered fills have no S4 counterfactual (as confidence.inputs)
                g = g[~g.cf_why.astype(str).eq("CONFIRMED_LIVE")]
            _alts = ("w1", "w2", "w4", "w8", "w12", "w24", "hi1", "hi2", "hi3", "hi4", "hi5", "hi6", "hi7")   # fb_combo.ALT_COLS (no import: it fetches)
            if not set(_alts) <= set(g.columns):   # a pre-474 fb_combo.csv would silently fall back to the too-easy 2-alternative test
                sys.exit("fb_combo.csv lacks the FB_CONFIRM search alternatives (w* / hi*) — re-run fb_combo.py")
            d = pd.DataFrame({"FB_CONFIRM": g.cf510 - g.now510, "FB +1/−4": g.cf14 - g.now510,
                              # 🎲 (474) the waits and higher sells the registration study also tried (fb_combo.ALT_COLS)
                              **{c: g[c] - g.now510 for c in g.columns if (c[:1] == "w" and c[1:].isdigit()) or (c[:2] == "hi" and c[2:].isdigit())}})
            res = LT.gain_line(d, g.o.dt.strftime("%Y-%m-%d").to_numpy(), "FB_CONFIRM")
        elif ln.startswith("EXIT "):
            g = FX.assign(o=pd.to_datetime(FX.k, format="mixed"))
            g = g if fz is None else g[(g.src == "MASTER") | (g.o < fz)]
            base = np.where(g.src == "B25", g.booked, g["stack"])
            d = pd.DataFrame({x: g[x] - base for x in EXITS})
            res = LT.gain_line(d, g.o.dt.strftime("%Y-%m-%d").to_numpy(), ln[5:])
        else:
            mA = (A.sl == sl) & (A.o < fz) if fz is not None else (A.sl == sl)
            parts = [M[M.sl == sl].assign(_z=LM[sl][ln].reindex(M.index[M.sl == sl]).fillna(False).astype(bool).values),
                     A[mA].assign(_z=LA[sl][ln].reindex(A.index[mA]).fillna(False).astype(bool).values)]
            g = pd.concat(parts, ignore_index=True, sort=False)
            g["day"] = g.o.dt.strftime("%Y-%m-%d")
            res = LT.block_line(g, g._z.to_numpy())
        LROWS.append(dict(sleeve=sl, line=ln, **res))
pd.DataFrame(LROWS).to_csv(os.path.join(SC, "luck.csv"), index=False)
LCALL = {(r["sleeve"], r["line"]): r["verdict"] for r in LROWS}
LS = ({}, set(), {})
for sl, combo in _best:
    keep = [ln for ln in combo.split(" + ") if LCALL.get((sl, ln)) != LT.LUCK]
    exs = [c[5:] for c in keep if c.startswith("EXIT ")]; ex = exs[0] if exs else None
    blk = [c for c in keep if c != "FB_CONFIRM" and not c.startswith("EXIT ")]
    LS[0].update(b25_after(sl, blk, "FB_CONFIRM" in keep, ex))
    if ex:
        LS[2].update({i: float(fxm[M.loc[i, "kk"]][ex]) for i in M.index[M.sl == sl]})
    z = pd.Series(False, index=M.index)
    for ln in blk:
        z |= LM[sl][ln].fillna(False).astype(bool)
    LS[1].update(set(M.index[z & (M.sl == sl)]))
print("\nLUCK CALLS: " + " · ".join(f"{sl}:{ln} {c}" for (sl, ln), c in LCALL.items()))
print(f"LUCK TOTAL B: B25 one size {tb:+.3f} → {b25_dcr(allm, LS[0]):+.3f} %/day · master intended size {mt_b:+.3f} → "
      f"{m_dcr(SLEEVES, frozenset(LS[1]), LS[2])[0]:+.3f}")
for nm, size in RESIZES:
    sz = {sl: f for sl, f in size.items() if f < 1 or not any(c == LT.LUCK for (s_, _l), c in LCALL.items() if s_ == sl)}   # a cap down always counts
    acc = b25_sized(LS[0], sz)
    print(f"LUCK {nm.split()[0]}: B25 total {acc_dcr(acc, pd.Series(True, index=acc.index)):+.3f} %/day"
          f" | master total {m_sized(LS[1], LS[2], sz):+.3f} · re-sized: {', '.join(sz) or 'none'}")
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
