#!/usr/bin/env python3
"""yr6 A/B (2026-10-13, operator-approved): what FRENZY_LONG + FRENZY_WIDE do to the REST of the book.

  A = yr6 / yr6a seed 1 (LONG / WIDE ON, frozen_config_yr6_01419d4.json, 83ecb8787a5b)
  B = yr6x / yr6ax seed 1 (LONG / WIDE OFF, frozen_config_yr6x_lwoff.json, 33b5784891bd) — same snapshot code_yr6_01419d4,
      same flags, same chunks, same cadence seed, same caches (10-11 chunk via a shadow cache, see ENGINE_REPLAY_YEAR_PLAN yr6x).
Both arms trimmed (scripts/yr5_fills_trimmed.py via scripts/yr6_tables.py) and RE-PRICED to today's config (scripts/yr6_reprice_today.py
run from a git-archive of HEAD 443a7b5: LONG / WIDE +1.5 at 2× / 1×, LITE +3 at 10×, WILLY 10×, momentum long × 0.5).

  cd <code_443a7b5 snapshot> && <venv>/bin/python "<repo>/scripts/study_yr6_lwoff_ab.py" \
      --a ENGINE_REPLAY_YR6LWA_TODAY_fills.csv --b ENGINE_REPLAY_YR6X_2Y_TODAY_fills.csv

Run from a code dir whose config.py / trading_config.json / services are TODAY's (the compounding book uses the engine's own
calculate_position_size). Matching: a fill is SHARED when (pair, direction, entry_strategy, opened_at) is identical in both arms;
leftovers with the same (pair, direction, entry_strategy) within ±--tol s are SHIFTED-shared (reported separately); the rest are
A-only / B-only. Outputs: reports/YR6_LWOFF_AB_{sleeves,decomp,lite_bonly,books}.csv + reports/YR6_LWOFF_AB_tables.txt.
Second computation: a stdlib-csv recount straight from the raw replay orders CSVs + meta windows (no pandas, own key code) — every
headline N / WR / avg % must match.
"""
import argparse, csv, glob, json, os, sys, warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
REPO = "/Users/guillearslanian/Downloads/NOFA AI"
REP = os.path.join(REPO, "reports")
RAW = os.path.join(REP, "backtest_cache", "replay")
ap = argparse.ArgumentParser()
ap.add_argument("--a", default="ENGINE_REPLAY_YR6LWA_TODAY_fills.csv")
ap.add_argument("--b", default="ENGINE_REPLAY_YR6X_2Y_TODAY_fills.csv")
ap.add_argument("--book", type=float, default=2900.0)
ap.add_argument("--draws", type=int, default=4000)
ap.add_argument("--tol", type=float, default=120.0)
A = ap.parse_args()
CODE = os.getcwd()
sys.path.insert(0, CODE)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./_yr6_lwoff_dummy.db"; os.environ["DEBUG"] = "false"
import logging  # noqa: E402
logging.disable(logging.WARNING)
import config  # noqa: E402
from services import trading_engine as te  # noqa: E402
assert os.path.abspath(os.path.dirname(config.__file__)) == os.path.abspath(CODE), "run from the today's-config code dir"
ENG = te.trading_engine; ENG._bnb_burn_rate = 0.0
INV = config.trading_config.investment; TH = config.trading_config.thresholds
FZ_LEV = {"FRENZY_LONG": float(TH.frenzy_long_lev_mult), "FRENZY_WIDE": float(TH.frenzy_wide_lev_mult),
          "FRENZY_LITE": float(TH.frenzy_lite_lev_mult), "FRENZY_WILLY": float(TH.frenzy_willy_lev_mult)}
rng = np.random.default_rng(20261013)
REGIMES = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
LW = ("FRENZY_LONG", "FRENZY_WIDE")


def load(fn):
    F = pd.read_csv(os.path.join(REP, fn), low_memory=False)
    F = F[F.seed == 1].copy()
    F["t"] = pd.to_datetime(F.t, format="mixed")
    F["tc"] = pd.to_datetime(F.closed_at.astype(str).str[:23].str.replace("T", " "), format="mixed")
    F["key"] = list(zip(F.pair, F.direction, F.entry_strategy.fillna("MOMENTUM"), F.opened_at.astype(str)))
    assert not F.key.duplicated().any()
    return F.reset_index(drop=True)


FA, FB = load(A.a), load(A.b)
assert not FB.sleeve.isin(LW).any(), "B arm has FRENZY_LONG / WIDE fills — wrong config?"

# ── matching ──────────────────────────────────────────────────────────────────────────────────────────────────────────
ka, kb = set(FA.key), set(FB.key)
FA["m"] = np.where(FA.key.isin(kb), "shared", "A-only"); FB["m"] = np.where(FB.key.isin(ka), "shared", "B-only")
# shifted: same pair/direction/strategy within ±tol s (greedy nearest, one-to-one)
FA["mate"] = -1; FB["mate"] = -1
gb = {k: g for k, g in FB[FB.m == "B-only"].groupby(["pair", "direction", FB.entry_strategy.fillna("MOMENTUM")])}
used = set()
for i, r in FA[FA.m == "A-only"].iterrows():
    g = gb.get((r.pair, r.direction, r.entry_strategy if isinstance(r.entry_strategy, str) else "MOMENTUM"))
    if g is None:
        continue
    dt = (g.t - r.t).abs().dt.total_seconds()
    dt = dt[(dt <= A.tol) & ~dt.index.isin(used)]
    if len(dt):
        j = dt.idxmin(); used.add(j)
        FA.at[i, "m"] = "shifted"; FB.at[j, "m"] = "shifted"; FA.at[i, "mate"] = j; FB.at[j, "mate"] = i
SH = FA[FA.m == "shared"].merge(FB[FB.m == "shared"][["key", "pct_new", "usd_new", "closed_at"]], on="key", suffixes=("_a", "_b"))
n_pct_diff = int(((SH.pct_new_a - SH.pct_new_b).abs() > 1e-6).sum())
n_close_diff = int((SH.closed_at_a.astype(str) != SH.closed_at_b.astype(str)).sum())

L = []
P = L.append
P(f"YR6 LONG/WIDE OFF A/B — seed 1 — A {A.a} ({len(FA)} fills) vs B {A.b} ({len(FB)} fills) · priced at {CODE}")
P(f"lev mult {FZ_LEV} · momentum long × {float(TH.momentum_long_lev_mult):g} · $ = fixed ${A.book:,.0f} book (usd_new) unless 'book'")
P(f"matching: shared {int((FA.m == 'shared').sum())} · shifted (±{A.tol:.0f} s) {int((FA.m == 'shifted').sum())} · A-only {int((FA.m == 'A-only').sum())} "
  f"· B-only {int((FB.m == 'B-only').sum())} · shared fills with a different pct_new {n_pct_diff} / different closed_at {n_close_diff}")


def stats(g):
    if not len(g):
        return dict(N=0, WR=np.nan, avg=np.nan, usd=0.0)
    return dict(N=len(g), WR=(g.pct_new > 0).mean() * 100, avg=g.pct_new.mean(), usd=g.usd_new.sum())


def day_boot(g, days, draws=None):
    """day-block bootstrap over the regime's calendar days (zero-fill days included): (avg % CI, Σ$ CI)."""
    draws = draws or A.draws
    d = g.groupby(g.t.dt.floor("D")).agg(s=("pct_new", "sum"), n=("pct_new", "size"), u=("usd_new", "sum")).reindex(days, fill_value=0)
    if d.n.sum() < 3:
        return (np.nan, np.nan), (np.nan, np.nan)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    s, n, u = d.s.values[idx].sum(1), d.n.values[idx].sum(1), d.u.values[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    return tuple(np.nanpercentile(m, [2.5, 97.5])), tuple(np.percentile(u, [2.5, 97.5]))


def regime_days(rg):
    lo, hi = {"2025 BULL": ("2025-01-01", "2026-01-04"), "2026-H1 BEAR": ("2026-01-04", "2026-07-01"),
              "2026-H2 BULL": ("2026-07-01", "2026-10-12")}[rg]
    return pd.date_range(lo, pd.Timestamp(hi) - pd.Timedelta(days=1), freq="D")


# ── 1. per sleeve × regime, A vs B ────────────────────────────────────────────────────────────────────────────────────────
rows = []
sleeves = sorted(set(FA.sleeve) | set(FB.sleeve))
for rg in REGIMES + ["ALL"]:
    for sl in sleeves + ["TOTAL", "TOTAL ex LONG/WIDE", "FRENZY family"]:
        def sel(F):
            q = F if rg == "ALL" else F[F.regime == rg]
            if sl == "TOTAL":
                return q
            if sl == "TOTAL ex LONG/WIDE":
                return q[~q.sleeve.isin(LW)]
            if sl == "FRENZY family":
                return q[q.sleeve.str.startswith("FRENZY")]
            return q[q.sleeve == sl]
        a, b = sel(FA), sel(FB)
        sa, sb = stats(a), stats(b)
        r = dict(regime=rg, sleeve=sl, **{f"A_{k}": v for k, v in sa.items()}, **{f"B_{k}": v for k, v in sb.items()}, d_usd=sb["usd"] - sa["usd"])
        for tag, F, mm in (("Aonly", a, "A-only"), ("Bonly", b, "B-only"), ("shiftA", a, "shifted"), ("shiftB", b, "shifted"), ("shared", a, "shared")):
            for k, v in stats(F[F.m == mm]).items():
                r[f"{tag}_{k}"] = v
        rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(os.path.join(REP, "YR6_LWOFF_AB_sleeves.csv"), index=False)


def fmt(n, wr, avg, usd):
    return f"{n:5.0f} · {wr:4.0f}% · {avg:+.3f} · ${usd:+8,.0f}" if n else f"{0:5d} ·    – ·    –   · ${0:+8,.0f}"


for rg in REGIMES + ["ALL"]:
    P(""); P(f"=== {rg}: per sleeve  A (LONG/WIDE ON)  vs  B (OFF) — N · WR · avg % · $ (fixed book, today's size) ===")
    P(f"{'sleeve':20} │ {'A':^34} │ {'B':^34} │ {'B−A $':>8} │ shared │ A-only N/avg/$ │ B-only N/avg/$ │ shifted A→B $")
    for r in S[S.regime == rg].itertuples():
        P(f"{r.sleeve:20} │ {fmt(r.A_N, r.A_WR, r.A_avg, r.A_usd)} │ {fmt(r.B_N, r.B_WR, r.B_avg, r.B_usd)} │ {r.d_usd:+8,.0f} │ {r.shared_N:6.0f} │ "
          f"{r.Aonly_N:4.0f} {r.Aonly_avg if r.Aonly_N else 0:+.2f} {r.Aonly_usd:+7,.0f} │ {r.Bonly_N:4.0f} {r.Bonly_avg if r.Bonly_N else 0:+.2f} {r.Bonly_usd:+7,.0f} │ "
          f"{r.shiftA_usd:+6,.0f}→{r.shiftB_usd:+6,.0f}")

# ── 2. FRENZY_LITE: B-only cohort, attribution to an A LONG/WIDE holding the coin, day-bootstrap ─────────────────────────────
LWA = FA[FA.sleeve.isin(LW)][["pair", "t", "tc", "sleeve"]]
lw_by_pair = {p: g for p, g in LWA.groupby("pair")}


def held_by_lw(r):
    g = lw_by_pair.get(r.pair)
    if g is None:
        return ""
    h = g[(g.t <= r.t) & (g.tc > r.t)]
    return h.sleeve.iloc[0] if len(h) else ""


LB = FB[(FB.sleeve == "FRENZY_LITE") & (FB.m == "B-only")].copy()
LB["held_in_A"] = [held_by_lw(r) for r in LB.itertuples()]
LA = FA[(FA.sleeve == "FRENZY_LITE") & (FA.m == "A-only")].copy()
LB.drop(columns=["key"]).to_csv(os.path.join(REP, "YR6_LWOFF_AB_lite_bonly.csv"), index=False)
P(""); P("=== FRENZY_LITE fills only in B (the ones LONG/WIDE kept out in A) — N · WR · avg % [95 % day CI] · $ [95 % day CI] ===")
D = []
for rg in REGIMES + ["ALL"]:
    days = (pd.DatetimeIndex(np.concatenate([regime_days(x) for x in REGIMES])) if rg == "ALL" else regime_days(rg))
    for sub, g0 in (("all B-only", LB), ("LONG/WIDE held the coin in A", LB[LB.held_in_A != ""]), ("other (path / slots)", LB[LB.held_in_A == ""]),
                    ("A-only LITE (lost when OFF)", LA)):
        g = g0 if rg == "ALL" else g0[g0.regime == rg]
        st = stats(g); (alo, ahi), (ulo, uhi) = day_boot(g, days)
        D.append(dict(regime=rg, cohort=sub, **st, avg_lo=alo, avg_hi=ahi, usd_lo=ulo, usd_hi=uhi))
        P(f"  {rg:13} {sub:30} {fmt(st['N'], st['WR'], st['avg'], st['usd'])}   avg CI [{alo:+.3f}, {ahi:+.3f}]  $ CI [{ulo:+,.0f}, {uhi:+,.0f}]")
pd.DataFrame(D).to_csv(os.path.join(REP, "YR6_LWOFF_AB_decomp.csv"), index=False)
_w = LB.loc[LB.pct_new.idxmin()] if len(LB) else None
if _w is not None:
    _r = LB.drop(index=LB.pct_new.idxmin())
    P(f"  sensitivity: without the single worst B-only LITE fill ({_w.pair} {str(_w.opened_at)[:16]} {_w.pct_new:+.2f} %, ${_w.usd_new:+,.0f}): "
      f"N {len(_r)} · WR {(_r.pct_new > 0).mean() * 100:.0f}% · avg {_r.pct_new.mean():+.3f} · ${_r.usd_new.sum():+,.0f}")
# today's LONG / WIDE take profit is +1.5 (the replay held them to +3): a holder whose replayed peak reached ≥ +1.5 BEFORE the LITE entry
# would have exited earlier at today's TP (first touch ≤ peak time) → that LITE is NOT blocked today; a holder that never reached +1.5
# exits identically today → still blocked. (Approximation: a LONG / WIDE re-entry on the same coin after a +1.5 exit is not modelled.)
_LWF = FA[FA.sleeve.isin(LW)].copy()
_LWF["tp_at"] = pd.to_datetime(_LWF.peak_reached_at.astype(str).str[:23].str.replace("T", " "), format="mixed", errors="coerce")


def _first_touch_before(pair, t0, t1, px):
    """True / False: did the price print ≥ px in [t0, t1)? (peak_reached_at is not stamped by the replay.) Real ticks (ticks_q / ticks,
    either cache) where present, else 1m bars fully inside the window (k1m_spike / k1m). None = no data."""
    root = os.path.join(REP, "backtest_cache_2025" if t0 < pd.Timestamp("2026-01-01") else "backtest_cache")
    a_ms, z_ms = int(t0.value // 1_000_000), int(t1.value // 1_000_000)
    ticks, seen_any = [], False
    for d in pd.date_range(t0.floor("D"), t1.floor("D"), freq="D"):
        for sub in ("ticks_q", "ticks"):
            f = os.path.join(root, sub, pair, f"{d:%Y-%m-%d}.npz")
            if os.path.exists(f):
                z = np.load(f); m = (z["t"] >= a_ms) & (z["t"] < z_ms); ticks.append(z["p"][m]); seen_any = True
                break
    if seen_any:
        return bool(any(len(x) and float(x.max()) >= px for x in ticks))
    for sub in ("k1m_spike", "k1m"):
        f = os.path.join(root, sub, f"{pair}.csv")
        if os.path.exists(f):
            k = pd.read_csv(f, usecols=["open_time", "h"])
            k = k[(k.open_time >= a_ms) & (k.open_time + 60_000 <= z_ms)]
            if len(k):
                return bool(k.h.max() >= px)
    return None


def _today_class(r):
    h = _LWF[(_LWF.pair == r.pair) & (_LWF.t <= r.t) & (_LWF.tc > r.t)]
    if not len(h):
        return ""
    h = h.iloc[0]
    if float(h.peak_pnl) < 1.5:
        return "still blocked today (holder never reached +1.5)"
    # TP basis is the fee-net pnl: gross threshold = +1.5 + the holder's own fee drag (gross move at exit − booked pct)
    fee = (float(h.exit_price) / float(h.entry_price) - 1) * 100 - float(h.pct)
    hit = _first_touch_before(r.pair, h.t, r.t, float(h.entry_price) * (1 + (1.5 + fee) / 100))
    if hit is None:
        return "unknown (no price data)"
    return "freed today (holder hit +1.5 before the LITE entry)" if hit else "still blocked today (holder hit +1.5 only after)"


LB["today_class"] = [_today_class(r) for r in LB.itertuples()]
LB.drop(columns=["key"]).to_csv(os.path.join(REP, "YR6_LWOFF_AB_lite_bonly.csv"), index=False)
P("  at TODAY's LONG / WIDE +1.5 TP (replay held them to +3):")
for cl, g in LB[LB.held_in_A != ""].groupby("today_class"):
    days = pd.DatetimeIndex(np.concatenate([regime_days(x) for x in REGIMES]))
    (alo, ahi), (ulo, uhi) = day_boot(g, days)
    P(f"    {cl:48} {fmt(len(g), (g.pct_new > 0).mean() * 100, g.pct_new.mean(), g.usd_new.sum())}  avg CI [{alo:+.3f}, {ahi:+.3f}]  $ CI [{ulo:+,.0f}, {uhi:+,.0f}]"
      f"  by regime {g.groupby('regime').size().to_dict()}")
for r in LB[LB.held_in_A != ""].sort_values("t").itertuples():
    P(f"      {str(r.opened_at)[:16]} {r.pair:12} LITE {r.pct_new:+6.2f} % ${r.usd_new:+5,.0f} · holder {r.held_in_A} · {r.today_class}")
_ex = LB[LB.held_in_A != ""].groupby("held_in_A").size().to_dict()
P(f"  holder in A of the B-only LITE coins: {_ex} · exit mix B-only LITE: {LB.exit_kind.value_counts().to_dict() if 'exit_kind' in LB else ''}")

# ── 3. the net question: is B − A (whole book) different from 0? day-bootstrap of the daily Δ$ ─────────────────────────────
P(""); P("=== NET: B − A per regime (fixed book $, today's size) — LONG/WIDE own $ in A vs everything else's Δ · day-bootstrap 95 % CI of Σ Δ$ ===")
NET = []
for rg in REGIMES + ["ALL"]:
    days = (pd.DatetimeIndex(np.concatenate([regime_days(x) for x in REGIMES])) if rg == "ALL" else regime_days(rg))
    a = FA if rg == "ALL" else FA[FA.regime == rg]; b = FB if rg == "ALL" else FB[FB.regime == rg]
    da = a.groupby(a.t.dt.floor("D")).usd_new.sum().reindex(days, fill_value=0)
    db = b.groupby(b.t.dt.floor("D")).usd_new.sum().reindex(days, fill_value=0)
    dlw = a[a.sleeve.isin(LW)].groupby(a[a.sleeve.isin(LW)].t.dt.floor("D")).usd_new.sum().reindex(days, fill_value=0)
    dd = (db - da).values; drest = (db - (da - dlw)).values
    idx = rng.integers(0, len(days), size=(A.draws, len(days)))
    lo, hi = np.percentile(dd[idx].sum(1), [2.5, 97.5]); plt0 = (dd[idx].sum(1) < 0).mean()
    lo2, hi2 = np.percentile(drest[idx].sum(1), [2.5, 97.5])
    r = dict(regime=rg, lw_own_usd=dlw.sum(), rest_delta_usd=drest.sum(), net_B_minus_A=dd.sum(), net_lo=lo, net_hi=hi, p_net_lt0=plt0,
             rest_lo=lo2, rest_hi=hi2, days=len(days))
    NET.append(r)
    P(f"  {rg:13} LONG/WIDE own $ in A {r['lw_own_usd']:+8,.0f} · Δ rest of book (B − A ex L/W) {r['rest_delta_usd']:+8,.0f} [{lo2:+,.0f}, {hi2:+,.0f}] "
      f"· NET B − A {r['net_B_minus_A']:+8,.0f} [{lo:+,.0f}, {hi:+,.0f}] · P(B<A) {plt0:.2f}")
pd.DataFrame(NET).to_csv(os.path.join(REP, "YR6_LWOFF_AB_net.csv"), index=False)

# ── 3b. decomposition: structural (fills that exist in one arm only) vs replay PATH NOISE (same entry, different exit) ──────
# A shared fill (identical pair / side / strategy / open timestamp) is the same trade in both arms; when its exit differs it is because
# the harness's seeded intra-minute tick ordering / 1 Hz monitor stream diverged after the arms' fill lists differed — not because
# LONG / WIDE changed anything about that trade. Its Δ is the replay's own noise floor (symmetric by construction); the CLEANED net
# holds every shared fill at its A outcome.
for F_ in (FB,):
    _hb = []
    for r in F_.itertuples():
        _hb.append(held_by_lw(r) if r.m == "B-only" else "")
    F_["held_in_A"] = _hb
P(""); P("=== DECOMPOSITION of B − A (fixed book $): ① LONG/WIDE own fills removed · ② structural rest (B-only − A-only + shifted Δ) · "
         "③ shared-fill re-exit noise · CLEANED net = ① + ② · day-bootstrap 95 % CI ===")
CHG = set(SH[(SH.pct_new_a - SH.pct_new_b).abs() > 1e-6].key)
DEC = []
for rg in REGIMES + ["ALL"]:
    days = (pd.DatetimeIndex(np.concatenate([regime_days(x) for x in REGIMES])) if rg == "ALL" else regime_days(rg))
    a = FA if rg == "ALL" else FA[FA.regime == rg]; b = FB if rg == "ALL" else FB[FB.regime == rg]

    def daily(g):
        return g.groupby(g.t.dt.floor("D")).usd_new.sum().reindex(days, fill_value=0).values
    lw = -daily(a[a.sleeve.isin(LW)])
    struct = (daily(b[b.m.isin(["B-only", "shifted"])]) - daily(a[a.m.isin(["A-only", "shifted"]) & ~a.sleeve.isin(LW)]))
    noise = daily(b[b.m == "shared"]) - daily(a[a.m == "shared"])
    idx = rng.integers(0, len(days), size=(A.draws, len(days)))
    ci = lambda v: tuple(np.percentile(v[idx].sum(1), [2.5, 97.5]))
    clean = lw + struct
    pc = (clean[idx].sum(1) < 0).mean()
    nsh = int((a.m == "shared").sum()); nchg = int(a[a.m == "shared"].key.isin(CHG).sum())
    r = dict(regime=rg, lw_removed=lw.sum(), structural_rest=struct.sum(), struct_lo=ci(struct)[0], struct_hi=ci(struct)[1], shared_noise=noise.sum(),
             noise_lo=ci(noise)[0], noise_hi=ci(noise)[1], clean_net=clean.sum(), clean_lo=ci(clean)[0], clean_hi=ci(clean)[1], p_clean_lt0=pc,
             raw_net=(lw + struct + noise).sum(), shared_n=nsh, shared_changed=nchg)
    DEC.append(r)
    P(f"  {rg:13} ① +{r['lw_removed']:,.0f} · ② {r['structural_rest']:+,.0f} [{r['struct_lo']:+,.0f}, {r['struct_hi']:+,.0f}] · ③ noise {r['shared_noise']:+,.0f} "
      f"[{r['noise_lo']:+,.0f}, {r['noise_hi']:+,.0f}] ({nchg}/{nsh} shared fills re-exited) · CLEANED NET B − A {r['clean_net']:+,.0f} "
      f"[{r['clean_lo']:+,.0f}, {r['clean_hi']:+,.0f}] P(B<A) {pc:.2f} · raw {r['raw_net']:+,.0f}")
pd.DataFrame(DEC).to_csv(os.path.join(REP, "YR6_LWOFF_AB_decomposition.csv"), index=False)
P("  ② by sleeve and cause (ALL regimes): B-only fills split by 'a LONG/WIDE held the coin in A at that moment' vs other (slots / path)")
for sl in sorted(set(FA.sleeve) | set(FB.sleeve)):
    if sl in LW:
        continue
    bo = FB[(FB.sleeve == sl) & (FB.m == "B-only")]; ao = FA[(FA.sleeve == sl) & (FA.m == "A-only")]
    sa = FA[(FA.sleeve == sl) & (FA.m == "shifted")].usd_new.sum(); sb = FB[(FB.sleeve == sl) & (FB.m == "shifted")].usd_new.sum()
    if not (len(bo) or len(ao) or abs(sb - sa) > 0.5):
        continue
    h = bo[bo.held_in_A != ""]; o = bo[bo.held_in_A == ""]
    P(f"    {sl:14} B-only coin-held {len(h):3d} ({h.pct_new.mean() if len(h) else 0:+.2f} %, ${h.usd_new.sum():+,.0f}) · B-only other {len(o):3d} "
      f"({o.pct_new.mean() if len(o) else 0:+.2f} %, ${o.usd_new.sum():+,.0f}) · A-only {len(ao):3d} ({ao.pct_new.mean() if len(ao) else 0:+.2f} %, "
      f"${ao.usd_new.sum():+,.0f}) · shifted Δ ${sb - sa:+,.0f} · ② ${bo.usd_new.sum() - ao.usd_new.sum() + sb - sa:+,.0f}")


# ── 4. shared compounding books (FRENZY-only and FULL), A vs B, today's sizing ────────────────────────────────────────────
def notional_at(equity, available, r, open_notional=0.0):
    s = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
    clm = FZ_LEV.get(s, float(r.cell_lev_multiplier or 1.0))
    inv, lev, _ = ENG.calculate_position_size(available, r.confidence, total_portfolio=equity, cell_multiplier=float(r.cell_multiplier or 1.0),
                                              cell_lev_multiplier=clm, multiplier_target="both",
                                              base_lev_multiplier=te.momentum_long_lev_factor(TH, s, r.direction))
    if inv <= 0 or lev <= 0:
        return 0.0, 0.0
    inv = ENG.momentum_long_resize("STUDY", inv, lev, available, r.confidence, equity, float(r.cell_multiplier or 1.0), clm, "both", s,
                                   r.direction, te.momentum_long_lev_factor(TH, s, r.direction))[0]   # ⚖️ DL 298: same notional, margin-scaled
    notional = inv * lev
    liq = float(INV.max_notional_pct_of_pair_volume or 0)
    vol = pd.to_numeric(pd.Series([r.get("entry_pair_volume_24h_usd")]), errors="coerce").iloc[0]
    if s.startswith("SPIKE") and INV.spike_lowvol_liq_cap_pct and vol and 0 < vol < INV.spike_lowvol_threshold_usd:
        liq = float(INV.spike_lowvol_liq_cap_pct)
    cap = liq / 100.0 * vol if (liq > 0 and vol and vol > 0) else None
    if INV.max_notional_hard_ceiling:
        cap = INV.max_notional_hard_ceiling if cap is None else min(cap, INV.max_notional_hard_ceiling)
    if cap is not None and notional > cap:
        notional = cap
    if INV.max_gross_leverage:
        notional = min(notional, max(0.0, equity * INV.max_gross_leverage - open_notional))
    inv = notional / lev
    if inv < INV.min_investment_size:
        return 0.0, 0.0
    return notional, inv


def book(F, bal0):
    F = F.sort_values("t").reset_index(drop=True)
    ev = []
    for i, r in F.iterrows():
        ev.append((r.t, 1, i)); ev.append((max(r.tc, r.t + pd.Timedelta(milliseconds=1)), 0, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    bal = bal0; openp = {}; curve = [(F.t.min(), bal0)]; skipped = 0
    for ts, kind, i in ev:
        r = F.loc[i]
        if kind == 1:
            om = sum(p[1] for p in openp.values()); on = sum(p[0] for p in openp.values())
            n, inv = notional_at(bal, bal - om, r, on)
            if n <= 0:
                skipped += 1; continue
            openp[i] = (n, inv)
        else:
            p = openp.pop(i, None)
            if p is not None:
                bal += float(r.pct_new) / 100.0 * p[0]; curve.append((ts, bal))
    c = pd.DataFrame(curve, columns=["ts", "bal"])
    mdd = ((c.bal / c.bal.cummax()) - 1).min() * 100
    days = max(1.0, (F.t.max() - F.t.min()).total_seconds() / 86400)
    dcr = ((bal / bal0) ** (1 / days) - 1) * 100 if bal > 0 else float("nan")
    return dict(end=round(bal, 2), mdd_pct=round(mdd, 1), dcr_pct=round(dcr, 4), skipped=skipped, n=len(F))


P(""); P(f"=== shared COMPOUNDING book from ${A.book:,.0f}, today's sizing (engine calculate_position_size, gross / liquidity caps, margin locked) ===")
BK = []
for per, lo, hi in (("2025", "2025-01-01", "2026-01-04"), ("2026", "2026-01-04", "2026-10-12"), ("2026-H1", "2026-01-04", "2026-07-01"),
                    ("2026-H2", "2026-07-01", "2026-10-12"), ("both years", "2025-01-01", "2026-10-12")):
    for scope in ("FRENZY only", "FULL"):
        out = {}
        for arm, F in (("A", FA), ("B", FB)):
            G = F[(F.t >= lo) & (F.t < hi)]
            if scope == "FRENZY only":
                G = G[G.sleeve.str.startswith("FRENZY")]
            out[arm] = book(G, A.book)
        BK.append(dict(period=per, scope=scope, **{f"A_{k}": v for k, v in out["A"].items()}, **{f"B_{k}": v for k, v in out["B"].items()}))
        P(f"  {per:10} {scope:12} A ${out['A']['end']:>10,.0f} (mdd {out['A']['mdd_pct']:.0f} %, {out['A']['dcr_pct']:+.3f} %/d, skip {out['A']['skipped']}) │ "
          f"B ${out['B']['end']:>10,.0f} (mdd {out['B']['mdd_pct']:.0f} %, {out['B']['dcr_pct']:+.3f} %/d, skip {out['B']['skipped']})")
pd.DataFrame(BK).to_csv(os.path.join(REP, "YR6_LWOFF_AB_books.csv"), index=False)

# ── 5. replay refusal counters (seed 1, meta filter_blocks, INCLUDING warm-up → upper bound) ──────────────────────────────
P(""); P("=== engine refusal counters, seed 1 (meta filter_blocks totals incl. warm-up = upper bound) ===")
for prefs, arm in ((("yr6a", "yr6"), "A"), (("yr6ax", "yr6x"), "B")):
    c = defaultdict(int)
    for pre in prefs:
        for f in glob.glob(os.path.join(RAW, f"{pre}_2*_s1_meta.json")):
            for row in (json.load(open(f)).get("filter_blocks") or {}).get("rows", []):
                nm = str(row.get("filter", ""))
                if "FRENZY" in nm and not row.get("is_variant"):
                    c[nm] += int(row.get("total") or 0)
    P(f"  {arm}: " + " · ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1])))

# ── 6. SECOND COMPUTATION: stdlib csv straight from the raw replay orders + meta windows ───────────────────────────────────
P(""); P("=== verification 2: stdlib recount from raw replay orders CSVs (own trim / key / sleeve code) vs the pandas pipeline ===")


def raw_fills(prefixes):
    out = {}
    for pre in prefixes:
        for mf in glob.glob(os.path.join(RAW, f"{pre}_2*_s1_meta.json")):
            tag = os.path.basename(mf)[:-10]
            if not tag.startswith(pre + "_2"):
                continue
            m = json.load(open(mf)); a, z = m["start_ms"], m["end_ms"]
            for r in csv.DictReader(open(os.path.join(RAW, f"{tag}_orders.csv"), newline="")):
                if r["status"] != "CLOSED" or "PROBE" in (r.get("cell_multiplier_source") or "") or r.get("entry_strategy") == "MANUAL":
                    continue
                ts = r["opened_at"][:23].replace("T", " ")
                t = datetime.strptime(ts + (".000" if len(ts) == 19 else ""), "%Y-%m-%d %H:%M:%S.%f")
                ms = (t - datetime(1970, 1, 1)).total_seconds() * 1000
                if not (a <= ms < z):
                    continue
                k = (r["pair"], r["direction"], r.get("entry_strategy") or "MOMENTUM", r["opened_at"])
                assert k not in out
                out[k] = r
    return out


RA, RB = raw_fills(("yr6a", "yr6")), raw_fills(("yr6ax", "yr6x"))
P(f"  raw trimmed fills A {len(RA)} (pipeline {len(FA)}) · B {len(RB)} (pipeline {len(FB)})")

def pct_today(r):   # re-price rule of yr6_reprice_today.py, written independently: LONG / WIDE first touch of +1.5 on the replayed peak
    s = r.get("entry_strategy") or ""; p = float(r["pnl_percentage"])
    if s in LW:
        pk = r.get("peak_pnl")
        if pk not in (None, "") and float(pk) >= 1.5:
            return 1.5
    return p


bo = [k for k in RB if k not in RA]
lite_bo = [pct_today(RB[k]) for k in bo if RB[k].get("entry_strategy") == "FRENZY_LITE"]
pipe = LB.pct_new
P(f"  B-only LITE: raw N {len(lite_bo)} · WR {np.mean([x > 0 for x in lite_bo]) * 100 if lite_bo else float('nan'):.1f}% · avg {np.mean(lite_bo) if lite_bo else float('nan'):+.4f}"
  f"  │ pipeline (exact + shifted-excluded) N {len(pipe)} · WR {(pipe > 0).mean() * 100:.1f}% · avg {pipe.mean():+.4f}"
  f"  (pipeline B-only excludes {int(((FB.sleeve == 'FRENZY_LITE') & (FB.m == 'shifted')).sum())} shifted LITE fills; pipeline B-only + shifted: N "
  f"{int(((FB.sleeve == 'FRENZY_LITE') & FB.m.isin(['B-only', 'shifted'])).sum())} avg {FB[(FB.sleeve == 'FRENZY_LITE') & FB.m.isin(['B-only', 'shifted'])].pct_new.mean():+.4f})")
for arm, R, F in (("A", RA, FA), ("B", RB, FB)):
    by = defaultdict(list)
    for k, r in R.items():
        by[r.get("entry_strategy") or "MOMENTUM"].append(pct_today(r))
    for s in sorted(by):
        v = by[s]; g = F[F.entry_strategy.fillna("MOMENTUM") == s].pct_new
        ok = len(v) == len(g) and abs(np.mean(v) - g.mean()) < 1e-6
        if not ok or s.startswith("FRENZY"):
            P(f"  {arm} {s:14} raw N {len(v):5d} avg {np.mean(v):+.4f} │ pipeline N {len(g):5d} avg {g.mean():+.4f} {'OK' if ok else 'MISMATCH'}")
# identity: B − A = B-only − A-only + shifted Δ + shared Δ
tot = FB.usd_new.sum() - FA.usd_new.sum()
parts = (FB[FB.m == "B-only"].usd_new.sum() - FA[FA.m == "A-only"].usd_new.sum() + FB[FB.m == "shifted"].usd_new.sum() - FA[FA.m == "shifted"].usd_new.sum()
         + FB[FB.m == "shared"].usd_new.sum() - FA[FA.m == "shared"].usd_new.sum())
P(f"  identity B − A ${tot:+,.2f} = B-only − A-only + shifted Δ + shared Δ ${parts:+,.2f}")
txt = "\n".join(L)
open(os.path.join(REP, "YR6_LWOFF_AB_tables.txt"), "w").write(txt + "\n")
print(txt)
