#!/usr/bin/env python3
"""yr6 2026 fills (replayed on config 01419d4) RE-PRICED to TODAY's config (b0f0682: DECISION_LOG 283 + 284).

  venv/bin/python scripts/yr6_reprice_today.py [--book 2900]

What changes vs the replay config (01419d4):
  · FRENZY_LONG / FRENZY_WIDE  exit +3 / −3 / 12 h → +1.5 / −3 / 12 h: first touch on the fill's own replayed peak_pnl (net, the TP's
    basis) — peak ≥ +1.5 books +1.5, else the replayed pct (a −3 stop / 12 h cap that never reached +1.5). EXACT for a lower TP with the
    same stop and cap (the walk is identical until +1.5 prints). Leverage 2× / 1× → 10× (lev mult 0.5).
  · FRENZY_LITE  exit unchanged (+3, DECISION_LOG 284); leverage 8× → 10× (0.4 → 0.5).
  · MOMENTUM LONG  leverage × 0.5 (momentum_long_lev_mult, DECISION_LOG 283) — % unchanged, $ via the engine's own sizing.
  · FRENZY_WILLY / shorts / flips / fades / SURGE / BULLRUN / BEARRUN unchanged (already today's).
Sizing: TODAY's services.trading_engine.calculate_position_size (repo HEAD config) with each fill's cell multipliers, the FRENZY sleeves'
today lev mult, and base_lev_multiplier = momentum_long_lev_factor — same ledger as scripts/yr6_tables.py (fixed book and compounding
shared book, gross / liquidity caps, margin locked while open).
NOT modelled (state it with every number): the replay's ENTRIES are the old config's — earlier +1.5 exits free FRENZY slots sooner (more
fills) and shorten the time an open LONG blocks LITE on the same coin; those second-order fills are absent.
Outputs: reports/ENGINE_REPLAY_YR6_TODAY_{sleeves,monthly,book}.csv + reports/ENGINE_REPLAY_YR6_TODAY.txt
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./_yr6_reprice_dummy.db"; os.environ["DEBUG"] = "false"
import logging  # noqa: E402
logging.disable(logging.WARNING)
import config  # noqa: E402
from services import trading_engine as te  # noqa: E402
from services.frenzy import frenzy_tp_for  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--book", type=float, default=2900.0); ap.add_argument("--draws", type=int, default=4000)
ap.add_argument("--fills", default="ENGINE_REPLAY_YR6_fills.csv"); ap.add_argument("--years", default="2026"); ap.add_argument("--tag", default="YR6_TODAY")
ap.add_argument("--no-book", action="store_true")
A = ap.parse_args()
ENG = te.trading_engine; ENG._bnb_burn_rate = 0.0
INV = config.trading_config.investment
TH = config.trading_config.thresholds
REP = os.path.join(ROOT, "reports")
rng = np.random.default_rng(20261012)
FZ_LEV = {"FRENZY_LONG": float(TH.frenzy_long_lev_mult), "FRENZY_WIDE": float(TH.frenzy_wide_lev_mult),
          "FRENZY_LITE": float(TH.frenzy_lite_lev_mult), "FRENZY_WILLY": float(TH.frenzy_willy_lev_mult)}
TP_NEW = {s: frenzy_tp_for(TH, s) for s in ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")}
OLD_TP = 3.0
assert float(TH.frenzy_lock_arm_pct or 0) == 0 and abs(float(TH.frenzy_stop_pct) - 3.0) < 1e-9 and int(TH.frenzy_max_hold_minutes) == 720


def reprice_pct(r):
    s = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
    if s in TP_NEW and TP_NEW[s] < OLD_TP - 1e-9:
        pk = pd.to_numeric(r.peak_pnl, errors="coerce")
        if pd.notna(pk) and pk >= TP_NEW[s]:
            return TP_NEW[s]
    elif s in TP_NEW and TP_NEW[s] > OLD_TP + 1e-9:
        raise SystemExit(f"{s}: a TP above the replayed +3 cannot be re-priced from the replay")
    return float(r.pct)


def size_args(r, today):
    s = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
    clm = float(r.cell_lev_multiplier or 1.0)
    base = 1.0
    if today:
        if s in FZ_LEV:
            clm = FZ_LEV[s]
        base = te.momentum_long_lev_factor(TH, s, r.direction)
    return float(r.cell_multiplier or 1.0), clm, base


def notional_at(equity, available, r, today, open_notional=0.0):
    cm, clm, base = size_args(r, today)
    inv, lev, _ = ENG.calculate_position_size(available, r.confidence, total_portfolio=equity, cell_multiplier=cm,
                                              cell_lev_multiplier=clm, multiplier_target="both", base_lev_multiplier=base)
    if inv <= 0 or lev <= 0:
        return 0.0, 0.0, lev
    if today:   # ⚖️ DL 298: a momentum long opens at the paired notional with a scaled margin (the engine's own step)
        s0 = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
        inv = ENG.momentum_long_resize("STUDY", inv, lev, available, r.confidence, equity, cm, clm, "both", s0, r.direction, base)[0]
    notional = inv * lev
    liq_pct = float(INV.max_notional_pct_of_pair_volume or 0)
    vol = pd.to_numeric(pd.Series([r.get("entry_pair_volume_24h_usd")]), errors="coerce").iloc[0]
    s = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
    if s.startswith("SPIKE") and INV.spike_lowvol_liq_cap_pct and vol and 0 < vol < INV.spike_lowvol_threshold_usd:
        liq_pct = float(INV.spike_lowvol_liq_cap_pct)
    cap = liq_pct / 100.0 * vol if (liq_pct > 0 and vol and vol > 0) else None
    if INV.max_notional_hard_ceiling:
        cap = INV.max_notional_hard_ceiling if cap is None else min(cap, INV.max_notional_hard_ceiling)
    if cap is not None and notional > cap:
        notional = cap
    if INV.max_gross_leverage:
        notional = min(notional, max(0.0, equity * INV.max_gross_leverage - open_notional))
    inv = notional / lev
    if inv < INV.min_investment_size:
        return 0.0, 0.0, lev
    return notional, inv, lev


def book(F, bal0, pcol, today):
    F = F.sort_values("t").reset_index(drop=True)
    F["closed"] = pd.to_datetime(F.closed_at.astype(str).str[:23].str.replace("T", " "), format="mixed")
    ev = []
    for i, r in F.iterrows():
        ev.append((r.t, 1, i)); ev.append((max(r.closed, r.t + pd.Timedelta(milliseconds=1)), 0, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    bal = bal0; openp = {}; curve = [(F.t.min(), bal0)]; skipped = 0
    for ts, kind, i in ev:
        r = F.loc[i]
        if kind == 1:
            om = sum(p[1] for p in openp.values()); on = sum(p[0] for p in openp.values())
            n, inv, lev = notional_at(bal, bal - om, r, today, on)
            if n <= 0:
                skipped += 1; continue
            openp[i] = (n, inv)
        else:
            p = openp.pop(i, None)
            if p is None:
                continue
            bal += float(r[pcol]) / 100.0 * p[0]; curve.append((ts, bal))
    c = pd.DataFrame(curve, columns=["ts", "bal"])
    mdd = ((c.bal / c.bal.cummax()) - 1).min() * 100
    daily = c.set_index("ts").bal.resample("D").last().ffill(); dret = daily.pct_change().dropna() * 100
    days = max(1.0, (F.t.max() - F.t.min()).total_seconds() / 86400)
    dcr = ((bal / bal0) ** (1 / days) - 1) * 100 if bal > 0 else float("nan")
    low20 = c[c.bal < 0.2 * bal0].ts.min()
    return dict(end=round(bal, 2), mdd_pct=round(mdd, 1), worst_day_pct=round(dret.min(), 1) if len(dret) else np.nan,
                dcr_pct=round(dcr, 3), below20=str(low20)[:10] if pd.notna(low20) else "never", skipped=skipped)


def day_ci(g, col):
    d = g.groupby(g.t.dt.floor("D"))[col].agg(["sum", "size"])
    if len(d) < 3:
        return (np.nan, np.nan)
    s, n = d["sum"].values, d["size"].values
    idx = rng.integers(0, len(d), size=(A.draws, len(d)))
    return tuple(np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5]))


F = pd.read_csv(os.path.join(REP, A.fills), low_memory=False)
F["t"] = pd.to_datetime(F.t, format="mixed")
F = F[F.year.astype(str).isin(A.years.split(","))].reset_index(drop=True)
F["regime"] = np.select([F.t < pd.Timestamp("2026-01-04"), F.t < pd.Timestamp("2026-07-01")], ["2025 BULL", "2026-H1 BEAR"], "2026-H2 BULL")
F["pct_new"] = [reprice_pct(r) for r in F.itertuples()]
F["nf_old"] = [notional_at(A.book, A.book, r, False)[0] for _, r in F.iterrows()]
F["nf_new"] = [notional_at(A.book, A.book, r, True)[0] for _, r in F.iterrows()]
F["usd_old"] = F.pct / 100 * F.nf_old
F["usd_new"] = F.pct_new / 100 * F.nf_new
seeds = sorted(F.seed.unique()); ns = len(seeds)
F.to_csv(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}_fills.csv"), index=False)   # per-fill re-priced rows (pct_new / usd_new / regime) for the W/L study
# two-way check: the "old" fixed-book $ must equal yr6_tables' usd_fixed (same snapshot sizing for unchanged sleeves)
_chk = F[~F.entry_strategy.isin(list(FZ_LEV)) & ~((F.entry_strategy == "MOMENTUM") & (F.direction == "LONG"))]
_dd = (_chk.usd_old - _chk.usd_fixed).abs()
print(f"check: unchanged-sleeve fixed-book $ vs yr6_tables usd_fixed — max |Δ| {_dd.max():.4f} on {len(_chk)} fills")

L = [f"yr6 RE-PRICED TO TODAY (443a7b5 · DECISION_LOG 283 + 284 + 286) — {len(F)} trimmed CLOSED fills · seeds {seeds} · "
     f"{F.t.min():%Y-%m-%d} → {F.t.max():%Y-%m-%d}",
     f"Today's FRENZY TP: LONG +{TP_NEW['FRENZY_LONG']:g} · WIDE +{TP_NEW['FRENZY_WIDE']:g} · LITE +{TP_NEW['FRENZY_LITE']:g} (stop −3, 12 h) · "
     f"lev mult {FZ_LEV} · momentum long × {float(TH.momentum_long_lev_mult):g}",
     "Entries are the replay config's (01419d4) — second-order fills from earlier +1.5 exits / shorter LONG-blocks-LITE are NOT modelled.",
     f"$ = fixed ${A.book:,.0f} book, engine sizing, no compounding (per seed). CI = day-bootstrap 95 % of avg %.", ""]
rows = []
L.append(f"{'sleeve':14} {'N/seed':>7} │ {'WR old':>6} {'avg% old':>9} {'$/seed old':>11} │ {'WR new':>6} {'avg% new':>9} {'95% CI new':>17} "
         f"{'$/seed new':>11} │ H1 / H2 avg% new")
for sl, g in F.groupby("sleeve"):
    lo, hi = day_ci(g, "pct_new")
    h1, h2 = g[g.t < "2026-05-01"].pct_new.mean(), g[g.t >= "2026-05-01"].pct_new.mean()
    row = dict(sleeve=sl, n_seed=len(g) / ns, wr_old=(g.pct > 0).mean() * 100, avg_old=g.pct.mean(), usd_old=g.usd_old.sum() / ns,
               wr_new=(g.pct_new > 0).mean() * 100, avg_new=g.pct_new.mean(), ci_lo=lo, ci_hi=hi, usd_new=g.usd_new.sum() / ns, h1=h1, h2=h2)
    rows.append(row)
    L.append(f"{sl:14} {row['n_seed']:7.0f} │ {row['wr_old']:5.1f}% {row['avg_old']:+9.3f} {row['usd_old']:+11,.0f} │ {row['wr_new']:5.1f}% "
             f"{row['avg_new']:+9.3f} [{lo:+.3f}, {hi:+.3f}] {row['usd_new']:+11,.0f} │ {h1:+.3f} / {h2:+.3f}")
S = pd.DataFrame(rows)
L.append(f"{'TOTAL':14} {len(F) / ns:7.0f} │ {'':6} {'':9} {S.usd_old.sum():+11,.0f} │ {'':6} {'':9} {'':17} {S.usd_new.sum():+11,.0f}")
S.to_csv(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}_sleeves.csv"), index=False)

# per regime (operator 2026-10-12: 2025 bull · 2026-H1 bear · 2026-H2 bull)
L += ["", "Per regime — N/seed · WR · avg % (95 % day CI) · $/seed at today's size (fixed book)"]
RG = []
for (sl, rg), g in F.groupby(["sleeve", "regime"]):
    lo, hi = day_ci(g, "pct_new")
    RG.append(dict(sleeve=sl, regime=rg, n_seed=len(g) / ns, wr=(g.pct_new > 0).mean() * 100, avg=g.pct_new.mean(), ci_lo=lo, ci_hi=hi, usd=g.usd_new.sum() / ns))
RG = pd.DataFrame(RG); RG.to_csv(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}_regimes.csv"), index=False)
for sl, g in RG.groupby("sleeve"):
    L.append(f"  {sl:14} " + " | ".join(f"{r.regime}: {r.n_seed:4.0f} · {r.wr:3.0f} % · {r.avg:+.3f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] · ${r.usd:+,.0f}" for r in g.itertuples()))
# FRENZY TP detail: how many LONG / WIDE fills the +1.5 changed, and which way
L += ["", "FRENZY_LONG / WIDE: what +1.5 did to each fill (pooled seeds)"]
for sl in ("FRENZY_LONG", "FRENZY_WIDE"):
    g = F[F.sleeve == sl]
    tp3 = g.pct >= 2.9; saved = (g.pct < 1.5) & (g.pct_new == 1.5); cut = tp3 & (g.pct_new == 1.5)
    L.append(f"  {sl}: {len(g)} fills · +3 winners cut to +1.5 {cut.sum()} ({-1.5 * cut.sum() + (3 - g[cut].pct).sum():+.1f} pts) · "
             f"losers/caps rescued to +1.5 {saved.sum()} ({(1.5 - g[saved].pct).sum():+.1f} pts) · untouched {len(g) - cut.sum() - saved.sum()}")

# monthly
M = F.groupby(["sleeve", F.t.dt.strftime("%Y-%m")]).agg(n=("pct", "size"), avg_old=("pct", "mean"), avg_new=("pct_new", "mean"),
                                                         usd_old=("usd_old", "sum"), usd_new=("usd_new", "sum")).reset_index()
M[["n", "usd_old", "usd_new"]] = M[["n", "usd_old", "usd_new"]] / ns
M.to_csv(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}_monthly.csv"), index=False)
L += ["", "Per month — avg % new (old) · $/seed new — per sleeve"]
piv = M.pivot(index="t", columns="sleeve", values="avg_new")
pio = M.pivot(index="t", columns="sleeve", values="avg_old")
piu = M.pivot(index="t", columns="sleeve", values="usd_new")
for sl in piv.columns:
    L.append(f"  {sl}: " + " · ".join(f"{m[5:]} {piv.loc[m, sl]:+.2f} ({pio.loc[m, sl]:+.2f}) ${piu.loc[m, sl]:+,.0f}"
                                         for m in piv.index if pd.notna(piv.loc[m, sl])))

if A.no_book:
    open(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0)
# shared compounding book per seed: full stack and FRENZY-only, old vs new
B = []
L += ["", f"Shared compounding book from ${A.book:,.0f} (per seed) — old (replay config) → new (today)"]
for scope, mask in (("FULL STACK", F.sleeve.notna()), ("FRENZY only", F.sleeve.str.startswith("FRENZY")),
                    ("FULL minus MOM-long & FLIP", ~F.sleeve.isin(["MOM-long", "FLIP-short"]))):
    for sd in seeds:
        G = F[mask & (F.seed == sd)]
        o = book(G, A.book, "pct", False); n = book(G, A.book, "pct_new", True)
        B.append(dict(scope=scope, seed=sd, **{f"old_{k}": v for k, v in o.items()}, **{f"new_{k}": v for k, v in n.items()}))
        L.append(f"  {scope:28} seed {sd}: ${o['end']:>10,.0f} (mdd {o['mdd_pct']:.0f} %, <20 % {o['below20']}) → ${n['end']:>10,.0f} "
                 f"(mdd {n['mdd_pct']:.0f} %, worst day {n['worst_day_pct']:.0f} %, {n['dcr_pct']:+.3f} %/day, <20 % {n['below20']})")
pd.DataFrame(B).to_csv(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}_book.csv"), index=False)
txt = "\n".join(L)
open(os.path.join(REP, f"ENGINE_REPLAY_{A.tag}.txt"), "w").write(txt + "\n")
print(txt)
