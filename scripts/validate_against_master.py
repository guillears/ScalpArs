#!/usr/bin/env python3
"""MASTER-BATCH VALIDATION GATE (operator Sep-28: "the MASTER BATCH is the source of truth … any change to the code, logic,
filters, calculations or trade reconstruction MUST be validated against the master batch before the work is complete").

Run BEFORE presenting any analysis / after any change to the analysis tooling. Exit code 1 on any FAIL.
  M1  ledger integrity      — every master fill: sign(pct used by the analyses) == sign(net $) (catches CF re-pricing slips)
  M2  ledger provenance     — every ledger fill is a kept, non-probe row of MASTER_POOL_stacked.csv (no invented fills)
  F1  feature grid          — every higher-TF bar of entry_feature_factory equals an INDEPENDENT groupby resample (catches
                              grid shifts / in-place mutation)
  F2  feature vs live stamps — rebuilt BTC 5m RSI and BTC 1h RSI vs the live-stamped entry_btc_rsi / entry_btc_rsi_1h on every
                              master fill: best-correlated time shift must be 0 and r ≥ 0.9 (catches clock / timezone / unit
                              misalignment against REAL trades)
  C1  calibration matches   — every MATCHED trade in reports/CALIBRATION_TABLE_trades.csv (if present): live pair/open/pct
                              equal the master row; backtest open within the match window
Usage: venv/bin/python scripts/validate_against_master.py
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import current_stack_ledger as LG
import entry_feature_factory as EF
M = LG.build()
sys.argv = _argv
FAILS = []
# Provenance reference for the research-artifact checks (C1 / X1 / PS1 — "is this row a REAL master fill?"): the ledger M plus the
# master fills today's stack refuses ONLY by the Oct-4 LONG_CHOP_BURST gate (stack 2026-10-04b). Those artifacts were built on the
# 10-04a ledger; a fill a later gate refuses is still a real as-traded fill, so it must not fail a provenance check.
_P0 = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
MPROV = pd.concat([M, _P0[(_P0.stack_block_reason == "LONG_CHOP_BURST") & (_P0.status == "CLOSED")]], ignore_index=True)


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


# M1 — the pct every analysis uses must agree in sign with the ledger's net $
pct = np.where(M.stack_block_reason.fillna("").str.contains("ARM040|LATE_ARM|FADE_SL"),
               M.stack_pnl / M.stack_ticket_scale.fillna(1) / pd.to_numeric(M.notional_value, errors="coerce") * 100, M.pnl_percentage)
bad = M[(np.sign(pct) != np.sign(M.net)) & (M.net.abs() > 0.5)]
check("M1 pct/net sign", len(bad) == 0, f"{len(bad)} of {len(M)} fills disagree"
      + ("" if not len(bad) else " → " + ", ".join(f"{p}@{str(o)[:16]}({r})" for p, o, r in zip(bad.pair, bad.opened_at, bad.stack_block_reason))))

# M2 — every ledger fill is a real, kept, non-probe master-pool fill (the ledger only SUBTRACTS further live gates)
P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
P = P[(P.status == "CLOSED") & (P.stack_keep.astype(str).str.lower().isin(["true", "1"]))
      & ~P.is_probe.astype(str).str.lower().isin(["true", "1"])]
key = lambda d: set(zip(d.opened_at.astype(str).str[:19].str.replace(" ", "T"), d.pair, d.direction))
extra = key(M) - key(P)
check("M2 ledger ⊆ pool (kept, non-probe)", len(extra) == 0,
      f"{len(M)} ledger fills, {len(extra)} not in the pool; pool kept non-probe {len(P)} → ledger gates drop {len(P) - len(M)}")

# F1 — higher-TF grid of the feature factory vs an independent resample
k = pd.read_csv(os.path.join(EF.K5, "BTCUSDT.csv"))
fr = EF._load("BTCUSDT")
for tf, ms in (("15m", 900_000), ("1h", 3_600_000), ("4h", 14_400_000)):
    g = k.assign(b=(k.open_time // ms) * ms).groupby("b").c.last()
    close_idx = pd.to_datetime(g.index + ms, unit="ms")
    ind = EF._ind(pd.DataFrame({"o": g, "h": g, "l": g, "c": g, "vol": g}).set_index(close_idx))   # rsi only needs c
    common = fr[tf].index.intersection(close_idx)
    d = (fr[tf].rsi.reindex(common) - ind.rsi.reindex(common)).abs()
    check(f"F1 {tf} grid", len(common) > 0.99 * len(close_idx) and d.max() < 1e-6,
          f"{len(common)}/{len(close_idx)} bars aligned, max RSI diff {d.max():.2e}")

# F2 — rebuilt BTC RSI vs LIVE stamps on every master fill (shift scan)
t = pd.to_datetime(M.opened_at.astype(str).str[:19])
for stamp, tf in (("entry_btc_rsi", "5m"), ("entry_btc_rsi_1h", "1h")):
    s = pd.to_numeric(M[stamp], errors="coerce")
    step = 5 if tf == "5m" else 60
    res = {}
    for sh in (-2, -1, 0, 1, 2):
        X = EF._asof({tf: fr[tf]}, t + pd.Timedelta(minutes=sh * step), "B")[f"B_{tf}_rsi"]
        res[sh] = pd.Series(X, index=M.index).corr(s)
    best = max(res, key=res.get)
    # live reads the FORMING bar → it sits between the last closed bar (shift 0) and the next (+1); both are correct alignments
    check(f"F2 {stamp} vs rebuilt {tf}", best in (0, 1) and res[best] >= 0.9,
          "r by shift(bars) " + " ".join(f"{k:+d}:{v:.3f}" for k, v in res.items()))

# C1 — calibration trade dump agrees with the master
cp = "reports/CALIBRATION_TABLE_trades.csv"
if os.path.exists(cp):
    C = pd.read_csv(cp)
    mt = C[C.kind.isin(["MATCHED", "MISSED"])]
    Mi = MPROV.assign(t=pd.to_datetime(MPROV.opened_at.astype(str).str[:19])).set_index(["pair", "t"])
    miss = 0
    for _, r in mt.iterrows():
        key = (r.pair, pd.Timestamp(r.live_open))
        if key not in Mi.index:
            miss += 1
    check("C1 calibration rows exist in master", miss == 0, f"{len(mt) - miss}/{len(mt)} live rows found")
    m = C[C.kind == "MATCHED"]
    dt = (pd.to_datetime(m.bt_open) - pd.to_datetime(m.live_open)).abs().dt.total_seconds()
    check("C1 matched open gap ≤ 20 min", bool((dt <= 1200).all()), f"max {dt.max():.0f}s")

# A1 — ML backtest audit (scripts/ml_backtest_audit.py, Oct-4): every LIVE row is a real as-traded fill (pair, open, pct) of the raw
# sources (BASE = COMBINED raw, B1 = BATCH1_FINAL, B2+ = master pool), every master-kept ML fill inside an audited window is present, and
# every MATCHED backtest row exists in its replay chunk's orders at that open with that pct.
import glob as _g
_raw = pd.concat([pd.read_csv("reports/COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv", low_memory=False),
                  pd.read_csv("reports/BATCH1_2026-07-11to31_orders_FINAL.csv", low_memory=False),
                  pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)], ignore_index=True)
_raw = _raw[_raw.status == "CLOSED"]
_rk = {(p, str(o)[:19].replace(" ", "T")): float(x) for p, o, x in zip(_raw.pair, _raw.opened_at, pd.to_numeric(_raw.pnl_percentage, errors="coerce"))}
for f in sorted(_g.glob("reports/ML_AUDIT_*_trades.csv")):
    A1 = pd.read_csv(f)
    lv = A1[A1.kind.isin(["MATCHED", "MISSED"])]
    bad = [(p, o) for p, o, x in zip(lv.pair, lv.live_open, lv.live_pct)
           if (p, str(o)[:19].replace(" ", "T")) not in _rk or abs(_rk[(p, str(o)[:19].replace(" ", "T"))] - x) > 1e-6]
    check(f"A1 {os.path.basename(f)} live rows = real fills", not bad, f"{len(lv) - len(bad)}/{len(lv)} found with equal pct" + (f" → {bad[:3]}" if bad else ""))
    mt = A1[A1.kind == "MATCHED"]
    miss = 0
    for tag, g in mt.groupby("bt_tag"):
        o = pd.read_csv(f"reports/backtest_cache/replay/{tag}_orders.csv", low_memory=False)
        ok = {(p, str(x)[:19]): float(y) for p, x, y in zip(o.pair, o.opened_at, pd.to_numeric(o.pnl_percentage, errors="coerce"))}
        miss += sum(1 for p, x, y in zip(g.pair, g.bt_open, g.bt_pct) if (p, str(x)[:19]) not in ok or abs(ok[(p, str(x)[:19])] - y) > 1e-6)
    check(f"A1 {os.path.basename(f)} matched bt rows exist", miss == 0, f"{len(mt) - miss}/{len(mt)} backtest rows found with equal pct")
    # master-kept ML (non-probe) fills inside the audited windows must all be audited (none silently dropped)
    win = A1.groupby("batch").live_open.agg(["min", "max"])
    Mk = M[(M.entry_strategy.fillna("MOMENTUM").astype(str) == "MOMENTUM") & (M.direction == "LONG") & ~M.is_probe.fillna(0).astype(bool)]
    Mk = Mk.assign(tt=pd.to_datetime(Mk.opened_at.astype(str).str[:19]))
    aud = set(zip(lv.pair, pd.to_datetime(lv.live_open)))
    drop = [(p, t) for p, t, e in zip(Mk.pair, Mk.tt, Mk.era) if e in win.index
            and pd.Timestamp(win.loc[e, "min"]) <= t <= pd.Timestamp(win.loc[e, "max"]) and (p, t) not in aud]
    check(f"A1 {os.path.basename(f)} master-kept ML inside windows audited", not drop, f"{len(drop)} missing" + (f" → {drop[:3]}" if drop else ""))

# Y1 — yr4 ML fills (scripts/yr4_ml_year_table.py): momentum LONG, CLOSED, non-probe, unique per seed, pct = net/notional
yp = "reports/ENGINE_REPLAY_YR4_ML_fills.csv"
if os.path.exists(yp):
    Y = pd.read_csv(yp, low_memory=False)
    okY = ((Y.entry_strategy.fillna("MOMENTUM") == "MOMENTUM") & (Y.direction == "LONG") & (Y.status == "CLOSED")
           & ~Y.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")).all()
    dup = int(Y.duplicated(["seed", "pair", "opened_at"]).sum())
    pdiff = (pd.to_numeric(Y.pnl, errors="coerce") / pd.to_numeric(Y.notional_value, errors="coerce") * 100 - Y.pct).abs().max()
    check("Y1 yr4 ML fills integrity", bool(okY) and dup == 0 and pdiff < 1e-6, f"{len(Y)} fills · dup {dup} · max |pct − net/notional| {pdiff:.1e}")

# X1 — ML exit replica (scripts/ml_exit_optimize.py, Oct-4): the AS-WAS tick replica must reproduce the LIVE master exits (bot exits:
# same reason ≥ 95 %, |mean Δpct| ≤ 0.02, corr ≥ 0.95) and the yr4 engine exits (same reason ≥ 95 %, |mean Δ| ≤ 0.01); every replica
# master key must be a real master ML fill with pct = pnl / notional.
xp, yp2 = "reports/ML_EXIT_OPT_variants_master.csv", "reports/ML_EXIT_OPT_variants_yr4.csv"
if os.path.exists(xp) and os.path.exists(yp2):
    def _rb(r):
        r = str(r)
        for a, b in (("STOP_LOSS_WIDE", "STOP"), ("STOP_LOSS", "STOP"), ("HARD_TP_LADDER", "LADDER"), ("HARD_TP", "FIXED_TP"),
                     ("RUNNER_TRAIL", "RUNNER"), ("TRAILING_STOP", "TRAIL_OLD")):
            r = r.replace(a, b)
        return r.split(" ")[0]
    WL = pd.read_csv("reports/ML_WATCHLIST_master_features.csv", low_memory=False)
    WL["key"] = "m|" + WL.opened_at.astype(str) + "|" + WL.pair
    _rk2 = {(p, str(o)[:19].replace(" ", "T")): x for p, o, x in zip(MPROV.pair, MPROV.opened_at, pd.to_numeric(MPROV.pnl_percentage, errors="coerce"))}
    badw = [k for k, p, o, x in zip(WL.key, WL.pair, WL.opened_at, WL.pnl_percentage)
            if (p, str(o)[:19].replace(" ", "T")) not in _rk2 or abs(_rk2[(p, str(o)[:19].replace(" ", "T"))] - x) > 1e-6]
    check("X1 exit-replica master inputs = real master fills", not badw, f"{len(WL) - len(badw)}/{len(WL)} found with equal pct")
    XR = pd.read_csv(xp); xa = XR[XR.variant == "ASWAS"].merge(WL[["key", "close_reason", "pnl_percentage"]], on="key")
    xa = xa[~xa.close_reason.map(_rb).isin(["MANUAL", "TRAIL_OLD"])]
    same = (xa.reason.map(_rb) == xa.close_reason.map(_rb)).mean(); md = (xa.pct - xa.pnl_percentage).mean()
    cr = np.corrcoef(xa.pct, xa.pnl_percentage)[0, 1]
    check("X1 AS-WAS replica vs live master exits", same >= 0.95 and abs(md) <= 0.02 and cr >= 0.95,
          f"{len(xa)} bot exits · same reason {same * 100:.1f}% · mean Δ {md:+.4f} · corr {cr:.3f}")
    YR = pd.read_csv(yp2); YB = YR[YR.variant == "BASE"]
    Yf = pd.read_csv("reports/ENGINE_REPLAY_YR4_ML_fills.csv", low_memory=False, usecols=["seed", "opened_at", "pair", "close_reason", "pct"])
    Yf["key"] = "y" + Yf.seed.astype(str) + "|" + Yf.opened_at.astype(str) + "|" + Yf.pair
    ya = YB.merge(Yf, on="key", suffixes=("", "_e"))
    same = (ya.reason.map(_rb) == ya.close_reason.map(_rb)).mean(); md = (ya.pct - ya.pct_e).mean()
    check("X1 replica vs yr4 engine exits", len(ya) == len(Yf) and same >= 0.95 and abs(md) <= 0.01,
          f"{len(ya)}/{len(Yf)} fills · same reason {same * 100:.1f}% · mean Δ {md:+.4f}")

# PS1 — pair-strength features (scripts/ml_pair_strength.py, Oct-4): every master row is a real master ML fill (pair, open, pct);
# REL 1h recomputed INDEPENDENTLY from the feature factory's 5m frames (PAIR ret12 − BTC ret12) equals the tool's value; the tool's
# rebuilt BTC distance from 5m EMA13 vs the LIVE stamp entry_btc_dist_from_ema13_pct: best-correlated bar shift must be 0, r ≥ 0.8
# (the live stamp uses the forming price, the rebuild the last closed bar → r ≈ 0.84 expected, not 1).
pp = "reports/ML_PAIR_STRENGTH_2026-10-04_master_features.csv"
if os.path.exists(pp):
    PS = pd.read_csv(pp, low_memory=False)
    WL2 = pd.read_csv("reports/ML_WATCHLIST_master_features.csv", low_memory=False)
    _rk3 = {(p, str(o)[:19].replace(" ", "T")): x for p, o, x in zip(MPROV.pair, MPROV.opened_at, pd.to_numeric(MPROV.pnl_percentage, errors="coerce"))}
    _wl = {(p, str(o)[:19].replace(" ", "T")): x for p, o, x in zip(WL2.pair, WL2.opened_at, WL2.pct)}
    badp = [(p, o) for p, o, x in zip(PS.pair, PS.opened_at, PS.pct)
            if (p, str(o)[:19].replace(" ", "T")) not in _rk3 or abs(_wl.get((p, str(o)[:19].replace(" ", "T")), np.nan) - x) > 1e-9]
    check("PS1 pair-strength master rows = real master ML fills", not badp and len(PS) == len(WL2),
          f"{len(PS) - len(badp)}/{len(PS)} found (watchlist {len(WL2)})" + (f" → {badp[:3]}" if badp else ""))
    tt = pd.to_datetime(PS.opened_at.astype(str).str[:19])
    bfr = EF._load("BTCUSDT")["5m"]; ib = bfr.index.searchsorted(tt.values, side="right") - 1
    rel = np.full(len(PS), np.nan)
    for pair, g in PS.groupby("pair"):
        fr = EF._load(pair)["5m"]; ip = fr.index.searchsorted(tt[g.index].values, side="right") - 1
        rel[g.index] = fr.ret12.values[ip] - bfr.ret12.values[ib[g.index]]
    dmax = np.nanmax(np.abs(rel - PS.rel1h.values))
    check("PS1 REL 1h vs independent factory recompute", dmax < 1e-6 and np.isfinite(rel).all(), f"max |Δ| {dmax:.1e} on {len(PS)} fills")
    kb = pd.read_csv("reports/backtest_cache/k5m_full/BTCUSDT.csv", usecols=["open_time", "c"])
    tb = (pd.to_datetime(kb.open_time, unit="ms") + pd.Timedelta("5min")).values; cb = kb.c.values
    e13 = pd.Series(cb).ewm(span=13, adjust=False).mean().values; jb = np.searchsorted(tb, tt.values, side="right") - 1
    stp = pd.to_numeric(WL2.set_index(WL2.pair + "|" + WL2.opened_at.astype(str)).entry_btc_dist_from_ema13_pct, errors="coerce").reindex(
        PS.pair + "|" + PS.opened_at.astype(str)).values
    ok = np.isfinite(stp)
    rs = {sh: np.corrcoef(stp[ok], ((cb[jb + sh] / e13[jb + sh] - 1) * 100)[ok])[0, 1] for sh in (-2, -1, 0, 1)}
    best = max(rs, key=rs.get); same = np.nanmax(np.abs(((cb[jb] / e13[jb] - 1) * 100) - PS.btc_d13.values))
    check("PS1 BTC-EMA13 rebuild vs live stamp", best == 0 and rs[0] >= 0.8 and same < 1e-6,
          f"best shift {best} (r {', '.join(f'{k:+d}:{v:.2f}' for k, v in rs.items())}) · tool = rebuild max |Δ| {same:.1e} · N {ok.sum()}")

# CB1 — 🌀👥 Oct-4 LONG_CHOP_BURST (DECISION_LOG 201), recomputed INDEPENDENTLY of the builder's pass: eff72 = live stamp, else a
# vectorised rolling rebuild (|C[t] − C[t−863]| / Σ|ΔC| over the 864 closed 5m bars before the fill, 3-dp truncated); the rebuild
# must agree with the live stamps (r ≥ 0.99). Burst = another kept, non-probe, non-MANUAL fill of the same era opened 0…120 s
# earlier that is not itself chop-burst refused. The builder's refused set must EQUAL this set (no extra, none missing).
P2 = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
_t = lambda d: pd.to_datetime(d.opened_at.astype(str).str[:19].str.replace("T", " "), errors="coerce")
P2["_o"] = _t(P2)
kc = pd.read_csv("reports/backtest_cache/k5m_full/BTCUSDT.csv", usecols=["open_time", "c"]).drop_duplicates("open_time").sort_values("open_time")
kc["close_t"] = pd.to_datetime(kc.open_time, unit="ms") + pd.Timedelta("5min")
_net = (kc.c - kc.c.shift(863)).abs(); _path = kc.c.diff().abs().rolling(863).sum()
kc["eff"] = np.floor(np.where(_path > 0, _net / _path, 0.0) * 1000 + 1e-9) / 1000
_j = np.searchsorted(kc.close_t.values, P2._o.values, side="right") - 1   # last bar CLOSED at/before the fill
_ok = (_j >= 999) & ((P2._o.values - kc.close_t.values[np.clip(_j, 0, None)]) <= np.timedelta64(15, "m"))
P2["_rb"] = np.where(_ok, kc.eff.values[np.clip(_j, 0, None)], np.nan)
_st = pd.to_numeric(P2.entry_btc_eff72, errors="coerce")
_v = _st.notna() & P2._rb.notna()
_r = np.corrcoef(_st[_v], P2._rb[_v])[0, 1] if _v.sum() > 2 else np.nan
P2["_eff"] = _st.where(_st.notna(), P2._rb)
_kept = P2.stack_keep.astype(str).str.lower().isin(["true", "1"]) | (P2.stack_block_reason == "LONG_CHOP_BURST")   # first-pass keep
_nb = _kept & ~P2.is_probe.astype(str).str.lower().isin(["true", "1"]) & (P2.entry_strategy.astype(str) != "MANUAL")
_ml = _nb & P2.entry_strategy.astype(str).str.startswith("MOMENTUM") & (P2.direction == "LONG")
exp = set()
for era, g in P2[_nb].sort_values(["_o"]).groupby("era", sort=False):
    for i, r in g.iterrows():
        if not _ml[i] or not (pd.notna(P2._eff[i]) and P2._eff[i] <= 0.007):
            continue
        dt = (r._o - g._o).dt.total_seconds()
        if ((dt >= 0) & (dt <= 120) & (g.index != i) & ~g.index.isin(list(exp))).any():
            exp.add(i)
got = set(P2.index[P2.stack_block_reason == "LONG_CHOP_BURST"])
check("CB1 LONG_CHOP_BURST refused set = independent recompute", got == exp and _r >= 0.99,
      f"builder {len(got)} · recompute {len(exp)} · eff72 rebuild vs live stamp r {_r:.3f} on {int(_v.sum())} rows · "
      + ", ".join(f"{P2.pair[i]}@{str(P2.opened_at[i])[:16]}({P2.era[i]})" for i in sorted(got | exp)))

print(f"\n{'ALL CHECKS PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
