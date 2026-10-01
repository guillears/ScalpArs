#!/usr/bin/env python3
"""Replay FIDELITY AUDIT for one sleeve (Sep-28, operator: "make 100% sure you do all tests").

Runs every check we know of in one pass and prints PASS / WARN / FAIL with numbers, replay vs LIVE CONTROL:
  1 integrity      — duplicates, fills outside chunk windows, probes, missing pnl
  2 stamp coverage — every entry_* column, replay vs live (a stamp that live records and the replay drops = FAIL)
  3 tick coverage  — share of fills whose pair-day has real trades (else exits run on a stepped 1m path)
  4 gate compliance— stamp-level checks of the sleeve's entry gates on BOTH replay and live-control fills; replay rate
                     ≫ live rate = replay bug; equal rates = stamp-definition artefact (stamp taken at open ≠ gate input)
  5 mechanics      — exit-reason mix, stop depth p10/median, avg win / avg loss, hold, maker share, fee per trade
  6 sizing         — notional ≤ liquidity cap; cell multipliers present
  7 recent window  — replay per-seed vs live (today's-rules basis = stack_pnl) over the near-current-code window
Live control = master pool stack-kept full-size fills (+ optional --batch CSVs), restricted to --live-from for the
rule-sensitive checks (near-current code). Exit code 0 always; read the verdict column.
  venv/bin/python scripts/replay_fidelity_audit.py --fills reports/backtest_cache/replay/year/yr2_report_fills.csv --sleeve MOM-long
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from types import SimpleNamespace
from services.trading_engine import long_heat_eval, long_megacap_block

ap = argparse.ArgumentParser()
ap.add_argument("--fills", required=True)
ap.add_argument("--sleeve", default="MOM-long")
ap.add_argument("--config", default="reports/backtest_cache/replay/frozen_config_5442bbc.json")
ap.add_argument("--batch", nargs="*", default=[])
ap.add_argument("--live-from", default="2026-09-16")
A = ap.parse_args()
CFG = json.load(open(A.config)); TH = CFG["thresholds"]; INV = CFG["investment"]
rows = []


def verdict(name, ok, warn, detail):
    rows.append((name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail))


def sleeve_of(s, d):
    s = str(s)
    if s.startswith("MOMENTUM"): return "MOM-long" if d == "LONG" else "MOM-short"
    if s.startswith("FLIP"): return "FLIP-short" if d == "SHORT" else "FLIP-long"
    return {"SPIKE_FADE": "Spike-Fade", "BULLRUN_LONG": "BullRun-Long", "BEARRUN_SHORT": "BearRun-Short"}.get(s, s)


R = pd.read_csv(A.fills, low_memory=False)
R["ts"] = pd.to_datetime(R.opened_at, format="mixed"); R["te"] = pd.to_datetime(R.closed_at, format="mixed")
R = R[R.sleeve == A.sleeve].copy()
M = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
M = M[(M.status == "CLOSED") & (M.stack_keep == True) & ~M.is_probe.fillna(False).astype(bool)].copy()
for b in A.batch:
    x = pd.read_csv(b, low_memory=False); x = x[x.status == "CLOSED"].copy(); x["era"] = "BATCH"; x["stack_pnl"] = x.pnl
    M = pd.concat([M, x], ignore_index=True)
M["sleeve"] = [sleeve_of(a, b) for a, b in zip(M.entry_strategy, M.direction)]
M = M[M.sleeve == A.sleeve].copy()
M["ts"] = pd.to_datetime(M.opened_at.astype(str).str[:19]); M["te"] = pd.to_datetime(M.closed_at.astype(str).str[:19], errors="coerce")
M = M.drop_duplicates(subset=["opened_at", "pair", "direction"], keep="last")
LC = M[M.ts >= A.live_from]                     # near-current-code live control
print(f"# Fidelity audit — {A.sleeve} · replay {len(R)} fills ({R.seed.nunique()} seeds) · live {len(M)} (control ≥{A.live_from}: {len(LC)})\n")

# 1 integrity
dup = R.duplicated(subset=["seed", "pair", "direction", "opened_at"]).sum()
probe = R.cell_multiplier_source.astype(str).str.contains("PROBE").sum()
verdict("1 integrity: duplicates / probes / missing pnl", dup == 0 and probe == 0 and R.pnl_percentage.notna().all(), False,
        f"dup {dup} · probe rows {probe} · missing pnl {R.pnl_percentage.isna().sum()}")

# 2 stamp coverage
bad, soft = [], []
for c in sorted(c for c in M.columns if c.startswith("entry_") and c in R.columns):
    lv, rp = M[c].notna().mean(), R[c].notna().mean()
    if lv >= 0.9 and rp < 0.5: bad.append(f"{c} live {lv:.0%} / replay {rp:.0%}")
    elif lv >= 0.9 and rp < 0.9: soft.append(f"{c} {lv:.0%}/{rp:.0%}")
verdict("2 stamp coverage (live ≥90% ⇒ replay must record it)", not bad, not bad and bool(soft),
        (f"MISSING {len(bad)}: " + "; ".join(bad[:8])) if bad else (f"partial: {'; '.join(soft[:6])}" if soft else "all live stamps recorded"))

# 3 tick coverage
has = [os.path.exists(f"reports/backtest_cache/ticks_q/{p}/{d:%Y-%m-%d}.npz") or os.path.exists(f"reports/backtest_cache/ticks/{p}/{d:%Y-%m-%d}.npz")
       for p, d in zip(R.pair, R.ts)]
R["ticks"] = has; cov = R.ticks.mean()
gap = R[~R.ticks].pnl_percentage.mean() - R[R.ticks].pnl_percentage.mean() if (~R.ticks).any() else 0.0
verdict("3 tick coverage of fill pair-days", cov >= 0.98, cov >= 0.85,
        f"{cov:.1%} covered · avg without ticks minus with {gap:+.3f}%/trade ({(~R.ticks).sum()} fills)")


# 4 gate compliance (stamp-level), replay vs live control
def n(d, c): return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)


def gates_long(d):
    door = d.cell_multiplier_source.astype(str).str.contains("CALM3D")
    rsi, adx = n(d, "entry_rsi"), n(d, "entry_adx")
    g = {
        "RSI in [min,max]": (rsi < TH["momentum_long_rsi_min"]) | (rsi > TH["momentum_long_rsi_max"]),
        "RSI (65,70] needs pADX≥rsiceil_door_adx_min": (rsi > 65) & (rsi <= 70) & (adx < TH["rsiceil_door_adx_min"]),
        "range position ≤ max": n(d, "entry_range_position") > TH["range_position_max_long"],
        "EMA5 stretch ≤ max": n(d, "entry_ema5_stretch") > TH["ema5_stretch_max_long"],
        "EMA gap 5-8 ≤ max": n(d, "entry_ema_gap_5_8") > TH["ema_gap_5_8_max_long"],
        "pair ATR in band": (n(d, "entry_atr_pct") < TH["pair_atr_min_long"]) | (n(d, "entry_atr_pct") > TH["pair_atr_max_long"]),
        "dist from EMA13 ≥ min": n(d, "entry_dist_from_ema13_pct") < TH["entry_dist_from_ema13_min_long"],
        "mega-cap rank": pd.Series([long_megacap_block(SimpleNamespace(long_megacap_rank_max=TH["long_megacap_rank_max"]), v)
                                    for v in d.get("entry_pair_rank", pd.Series(index=d.index))], index=d.index),
        "heat block (stamped 30d)": pd.Series([long_heat_eval(SimpleNamespace(**{k: TH[k] for k in TH if k.startswith("long_heat")}),
                                                              a, b_, c, e)[1] for a, b_, c, e in
                                               zip(n(d, "entry_btc_ema20_slope"), n(d, "entry_btc_rsi_prev"), n(d, "entry_bull_pct"),
                                                   n(d, "entry_btc_off30d_high_pct"))], index=d.index),
        "CALM3D door: BTC ATR ≤ max": door & (n(d, "entry_btc_atr_pct") > TH["nonexp_calm3d_btc_atr_max"]),
        "CALM3D door: stretch ≤ max": door & (n(d, "entry_ema5_stretch") > TH["nonexp_calm3d_max_stretch"]),
        "CALM3D door: +DI / pADX floors": door & ((n(d, "entry_pos_di") < TH["nonexp_calm3d_min_pos_di"]) | (adx < TH["nonexp_calm3d_min_pair_adx"])),
        "CALM3D door: regime": door & ~d.entry_btc_regime.astype(str).isin(TH["nonexp_calm3d_regimes"].split(",")),
        "blacklist": d.pair.isin(str(CFG.get("pair_blacklist", "")).split(",")),
    }
    return g


if A.sleeve == "MOM-long":
    GR, GL = gates_long(R), gates_long(LC)
    worst = []
    for k in GR:
        rr, ll = GR[k].fillna(False).mean(), GL[k].fillna(False).mean()
        worst.append((rr - ll, k, rr, ll))
        ok = rr <= max(ll * 1.5, 0.01); warn = rr <= max(ll * 3, 0.03)
        verdict(f"4 gate: {k}", ok, warn, f"replay {rr:.1%} ({int(GR[k].fillna(False).sum())}) vs live-control {ll:.1%} ({int(GL[k].fillna(False).sum())})")

# 5 mechanics
def cr(x): return x.close_reason.astype(str).str.replace(r"[ _]L\d+$", "", regex=True)
def mech(x):
    w, l = x[x.pnl_percentage > 0], x[x.pnl_percentage <= 0]
    st = x[cr(x).str.contains("STOP_LOSS")]
    fee = (x.total_fee / x.notional_value * 100) if "total_fee" in x else pd.Series(dtype=float)
    return dict(WR=(x.pnl_percentage > 0).mean() * 100, win=w.pnl_percentage.mean(), loss=l.pnl_percentage.mean(),
                stop_share=len(st) / max(1, len(x)) * 100, stop_med=st.pnl_percentage.median(), stop_p10=st.pnl_percentage.quantile(.1),
                hold=((x.te - x.ts).dt.total_seconds() / 60).median(), maker=(x.entry_order_type == "MAKER").mean() * 100,
                fee=pd.to_numeric(fee, errors="coerce").mean(), peak=pd.to_numeric(x.peak_pnl, errors="coerce")[x.pnl_percentage > 0].mean())
mr, ml = mech(R), mech(M)
for k, tol in [("stop_med", 0.08), ("stop_p10", 0.25), ("loss", 0.15), ("win", 0.15), ("hold", 6), ("maker", 15), ("fee", 0.02)]:
    d = abs(mr[k] - ml[k])
    verdict(f"5 mechanics: {k}", d <= tol, d <= 2 * tol, f"replay {mr[k]:+.3f} vs live {ml[k]:+.3f} (|Δ| {d:.3f}, tol {tol})")
verdict("5 mechanics: stop share (selection-sensitive)", True, True, f"replay {mr['stop_share']:.0f}% vs live {ml['stop_share']:.0f}% — "
        "compare only with the matched-trade check below")

# 5b matched trades (same pair ±3 min): identical-trade parity
mm = []
for r in M.itertuples():
    x = R[(R.pair == r.pair) & ((R.ts - r.ts).abs() <= pd.Timedelta("3min"))]
    for y in x.itertuples():
        mm.append((r.pnl_percentage, y.pnl_percentage, str(r.close_reason).startswith("STOP"), str(y.close_reason).startswith("STOP"), r.ts >= pd.Timestamp(A.live_from)))
D = pd.DataFrame(mm, columns=["live", "rep", "ls", "rs", "recent"])
if len(D):
    dd = (D.rep - D.live).mean(); rec = D[D.recent]
    verdict("5b matched trades: avg replay − live", abs(dd) <= 0.10, abs(dd) <= 0.2,
            f"n {len(D)} · Δ {dd:+.3f}%/trade · sign agree {((D.live > 0) == (D.rep > 0)).mean():.0%} · stop share live {D.ls.mean():.0%} / replay {D.rs.mean():.0%}"
            + (f" · recent-code n {len(rec)} Δ {(rec.rep - rec.live).mean():+.3f}" if len(rec) else ""))

# 6 sizing
capn = n(R, "entry_liquidity_cap_notional"); over = (R.notional_value > capn * 1.001) & capn.notna()
verdict("6 sizing: notional ≤ liquidity cap", over.sum() == 0, over.mean() < 0.01, f"{over.sum()} fills above their cap")
verdict("6 sizing: cell multiplier stamped", R.cell_multiplier.notna().mean() > 0.99, False, f"{R.cell_multiplier.notna().mean():.1%}")

# 7 recent window, today's-rules basis
W = M[M.ts >= A.live_from].groupby("era").ts.agg(["min", "max"])
RR = R[[((W["min"] <= t) & (t <= W["max"])).any() for t in R.ts]]
if len(LC) and len(RR):
    lp = (LC.pnl_percentage * LC.stack_pnl / LC.pnl.replace(0, np.nan)).fillna(LC.pnl_percentage)
    lp = lp.where(~LC.stack_block_reason.fillna("").str.contains("CF_FADE_CAP05"), LC.pnl_percentage)   # a SIZE re-price: pct unchanged
    per = RR.groupby("seed").pnl_percentage.mean()
    inside = per.min() - 0.1 <= lp.mean() <= per.max() + 0.1
    verdict("7 recent window: live (today's rules) inside replay seed range ±0.1", inside, True,
            f"live {len(LC)} · {lp.mean():+.3f}% vs replay/seed {len(RR) / max(1, R.seed.nunique()):.1f} · seeds {per.min():+.3f}…{per.max():+.3f}")

w = max(len(r[0]) for r in rows)
for name, v, det in rows:
    print(f"{v:4}  {name:<{w}}  {det}")
print(f"\nSUMMARY: {sum(r[1] == 'PASS' for r in rows)} PASS · {sum(r[1] == 'WARN' for r in rows)} WARN · {sum(r[1] == 'FAIL' for r in rows)} FAIL")
