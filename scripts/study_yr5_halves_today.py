#!/usr/bin/env python3
"""📊 Oct-8 research — yr5 backtest by strategy, H1 / H2 / full year, re-priced at TODAY's stack (live config 2026-10-08, master STACK 2026-10-08b).

Read-only over: yr5 engine replay (code 181131e, 3 seeds, warm-up trimmed by scripts/yr5_fills_trimmed.py), the FRENZY engine-parity
cohort (above_streak join), HEAT_YR5_SIGNALS_priced.csv (heat re-admits), the FRENZY_LITE study cohort (scratch lite_streak/kept.pkl, the
724-fill "12 closes, judged once" universe), the WILLY re-price (scripts/study_yr5_halves_willy.py output) and the vol/mcap study's
FRENZY TODAY tick cohort (cross-check). BTC regime rebuilds: scripts/study_yr5_halves_btc.py. Sizing: scripts/build_master_pool.py
today_size_rule (frozen 2026-10-08b constants) + the frozen sleeve sizes.

$ convention (same as reports/YR5_DAILY_COMPOUND_BY_STRATEGY_2026-10-06.md): a FLAT $3,000 book per strategy, no compounding;
$ = 3000 × 0.24375 (equal split, 4 slots, 2.5 % fee reserve) × invest mult × leverage × fill ratio × pnl % / 100.
Split: H1 = [2026-01-04, 2026-05-20) 136 days · H2 = [2026-05-20, 2026-10-04) 137 days, by opened_at (UTC).
Usage: venv/bin/python scripts/study_yr5_halves_today.py [--oct6]   (--oct6 = reproduce the Oct-6 table's pricing, parity check)"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import yr5_fills_trimmed as YT                                   # noqa: E402
from build_master_pool import today_size_rule                    # noqa: E402
import study_yr5_halves_btc as BT                                # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
REP = os.path.join(ROOT, "reports")
BOOK, SLOT = 3000.0, 0.975 / 4
T0, TS, T1 = pd.Timestamp("2026-01-04"), pd.Timestamp("2026-05-20"), pd.Timestamp("2026-10-04")
RES_SCHED = [(10000, 8000), (25000, 17500), (50000, 27500), (100000, 40000), (150000, 50000), (250000, 70000), (500000, 100000)]
LEV = lambda m: max(1, int(round(20 * m)))                       # engine: round(20 × lev mult)
FADE_CAP_PCT, FADE_CEIL = 0.005, 500_000.0
ATR_NEW, ATR_OLD, HG_STREAK, TP = 3.0, 2.5, 12, 3.0
SLEEVES = ["MOM_LONG", "MOM_SHORT", "FLIP_SHORT", "SPIKE_FADE", "BULLRUN", "BEARRUN", "FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE",
           "FRENZY_WILLY", "SURGE_LONG", "SURGE_SHORT"]
NAME = {"MOM-long": "MOM_LONG", "MOM-short": "MOM_SHORT", "FLIP-short": "FLIP_SHORT", "Spike-Fade": "SPIKE_FADE", "BullRun-Long": "BULLRUN",
        "BearRun-Short": "BEARRUN", "FRENZY_LONG": "FRENZY_LONG", "FRENZY_WIDE": "FRENZY_WIDE", "SURGE_LONG": "SURGE_LONG", "SURGE_SHORT": "SURGE_SHORT"}


def ms(ts):
    return pd.to_datetime(pd.Series(ts).astype(str).str[:23].str.replace("T", " "), format="mixed").values.astype("datetime64[ms]").astype("int64")


def oct6_fill_ratio():
    """the Oct-6 table's fill ratio (replay shrink vs its own chunk book), keyed (tag, pair, opened_at)."""
    RD = os.path.join(REP, "backtest_cache", "replay"); out = {}
    for f in sorted(glob.glob(RD + "/yr5_*_orders.csv")):
        tag = os.path.basename(f)[:-11]
        rs = pd.read_csv(f, low_memory=False)
        cl = np.sort(ms(rs.closed_at)); pnl = (rs.pnl.astype(float) + rs.total_fee.astype(float)).values[np.argsort(ms(rs.closed_at))]
        cum = np.cumsum(pnl)
        o = ms(rs.opened_at)
        k = np.searchsorted(cl, o, side="right")
        eq = 5000 + np.where(k > 0, cum[np.maximum(k - 1, 0)], 0.0)
        tgt = np.array([max([t for b, t in RES_SCHED if e >= b], default=np.nan) for e in eq])
        base = np.where(np.isnan(tgt), eq, tgt) * SLOT
        fr = np.minimum(1.0, rs.investment.astype(float).values / base / rs.cell_multiplier.astype(float).values)
        for p, oa, v in zip(rs.pair, rs.opened_at.astype(str), fr):
            out[(tag, p, oa)] = v
    return out


def boot_day(v, d, w=None, B=4000, seed=7):
    """day-clustered bootstrap 95 % CI of the (weighted) mean."""
    if len(v) < 2:
        return np.nan, np.nan
    w = np.ones(len(v)) if w is None else np.asarray(w, float)
    g = pd.DataFrame({"s": np.asarray(v, float) * w, "w": w, "d": d}).groupby("d")[["s", "w"]].sum()
    s, c, k = g.s.values, g.w.values, len(g)
    rng = np.random.default_rng(seed); i = rng.integers(0, k, (B, k))
    m = s[i].sum(1) / c[i].sum(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


# ───────────────────────────────────────────── yr5 replay sleeves ─────────────────────────────────────────────
def build_replay(oct6=False):
    F = YT.load()
    F["S"] = F.sleeve.map(NAME)
    F["tms"] = F.t.values.astype("datetime64[ms]").astype("int64")
    fr = oct6_fill_ratio()
    F["fr"] = [fr.get((tg, p, str(o)), np.nan) for tg, p, o in zip(F.tag, F.pair, F.opened_at.astype(str))]
    assert F.fr.notna().all(), "fill ratio join failed"
    F["pk"] = pd.to_numeric(F.peak_pnl, errors="coerce")
    cm = pd.to_numeric(F.cell_multiplier, errors="coerce").fillna(1.0); clm = pd.to_numeric(F.cell_lev_multiplier, errors="coerce").fillna(1.0)
    F["note"] = ""; F["keep"] = True; F["why"] = ""
    F["pct_today"] = F.pct
    if oct6:   # the Oct-6 table's rules, for parity: sizing only, frozen entries/exits
        inv = cm.copy(); src = F.cell_multiplier_source.fillna("").astype(str)
        inv[(F.S == "FLIP_SHORT") & (cm > 1) & (src.str.contains("NEGDI15") | src.str.contains("TG_SHALLOW"))] = 1.0
        inv[(F.S == "MOM_LONG") & src.str.contains("UNMATCHED") & (cm >= 2.0)] = 1.5
        lev = np.where(F.S == "BEARRUN", 1, [LEV(x) for x in clm])
        F["usd"] = BOOK * SLOT * inv * F.fr * lev * F.pct / 100
        return F
    # ── FRENZY family: WIDE re-coded under the 3.0 ATR cap; bearish-day block; fixed +3 / −3 ──
    atr = pd.to_numeric(F.entry_atr_pct, errors="coerce"); br = pd.to_numeric(F.entry_frenzy_bar_ret_pct, errors="coerce")
    w = F.S == "FRENZY_WIDE"
    moved = w & (br <= 0) & (atr > ATR_OLD) & (atr <= ATR_NEW)
    F.loc[moved, "S"] = "FRENZY_LONG"; F.loc[moved, "note"] = "WIDE→FRENZY_LONG (ATR 2.5–3.0 red, cap 3.0)"
    w = F.S == "FRENZY_WIDE"
    # WIDE hold-green: GREEN_BAR (ATR ≤ 3.0, green) ∧ above_streak > 12 (engine cohort join; unknown → refused, fail-closed like the builder)
    C = pd.read_csv(os.path.join(REP, "FRENZY_ENGINE_COHORT_2026-10-05.csv"))[["pair", "t_signal_close", "above_streak", "atr"]].sort_values("t_signal_close")
    Z = F[F.S.isin(["FRENZY_LONG", "FRENZY_WIDE"])][["pair", "tms"]].reset_index().sort_values("tms")
    J = pd.merge_asof(Z, C, left_on="tms", right_on="t_signal_close", by="pair", direction="backward", tolerance=15 * 60 * 1000).set_index("index")
    F["above_streak"] = J.above_streak.reindex(F.index)
    F["streak_atr_match"] = (np.abs(J.atr.reindex(F.index) - atr) < 0.01)
    green = br > 0
    hg_code = np.where(atr > ATR_NEW, "ATR_HIGH", np.where(green, "GREEN_BAR", "RED"))
    refuse = w & ~((hg_code == "GREEN_BAR") & (F.above_streak > HG_STREAK))
    F.loc[refuse, "keep"] = False
    F.loc[refuse & (atr > ATR_NEW), "why"] = "WIDE_ATR_HIGH(>3.0)"
    F.loc[refuse & (atr <= ATR_NEW) & F.above_streak.isna() & green, "why"] = "WIDE_STREAK_UNKNOWN(fail-closed)"
    F.loc[refuse & (atr <= ATR_NEW) & F.above_streak.notna() & green, "why"] = "WIDE_RECLAIM(streak≤12)"
    F.loc[refuse & (F.why == ""), "why"] = "WIDE_OTHER"
    fz = F.S.isin(["FRENZY_LONG", "FRENZY_WIDE"])
    bear = np.array([BT.bearish(a, b) is True for a, b in zip(pd.to_numeric(F.entry_btc_1d_ret_pct, errors="coerce"), pd.to_numeric(F.entry_btc_trend_gap_pct, errors="coerce"))])
    blk = fz & F.keep & bear
    F.loc[blk, "keep"] = False; F.loc[blk, "why"] = "FRENZY_BEARISH_DAY"
    F.loc[fz, "pct_today"] = np.where(F.loc[fz, "pk"] >= TP, TP, F.loc[fz, "pct"])
    # ── sizing ──
    lev = pd.Series(20.0, index=F.index); inv = pd.Series(1.0, index=F.index)
    strong = (pd.to_numeric(F.entry_frenzy_adx_delta, errors="coerce") > 0) & (pd.to_numeric(F.entry_frenzy_di_spread, errors="coerce") > 0)
    lev[F.S == "FRENZY_LONG"] = np.where(strong[F.S == "FRENZY_LONG"], LEV(0.5), LEV(0.32))
    lev[F.S == "FRENZY_WIDE"] = LEV(0.2); lev[F.S == "BEARRUN"] = LEV(0.25)
    lev[F.S.isin(["BULLRUN", "SURGE_LONG", "SURGE_SHORT"])] = 20.0
    cellset = F.S.isin(["MOM_LONG", "MOM_SHORT", "FLIP_SHORT"])
    fac = [today_size_rule(s, d, src, c, g, sl, pv, lv, None)[0] if cs else 1.0
           for s, d, src, c, g, sl, pv, lv, cs in zip(F.entry_strategy.fillna("MOMENTUM"), F.direction, F.cell_multiplier_source, cm,
                                                      F.entry_global_volume_ratio, F.entry_btc_ema20_slope, F.entry_pair_volume_ratio, clm, cellset)]
    F["size_factor"] = fac
    inv[cellset] = (cm * clm * F.size_factor)[cellset]           # today's (invest × lev-mult) relative to a 1× 20× cell
    usd = BOOK * SLOT * inv * lev * F.fr * F.pct_today / 100
    # SPIKE_FADE: today's ticket on a $3k book = min(desired, 0.5 % × 24 h volume, ceiling); desired = 3000 × 0.24375 × 2 × 20
    fd = F.S == "SPIKE_FADE"
    want = BOOK * SLOT * cm * 20
    tick = np.minimum(np.minimum(want, FADE_CAP_PCT * pd.to_numeric(F.entry_pair_volume_24h_usd, errors="coerce")), FADE_CEIL)
    usd[fd] = (tick * F.pct / 100)[fd]
    F["fade_ticket_ratio"] = np.where(fd, tick / want, np.nan)
    F["usd"] = np.where(F.keep, usd, 0.0)
    F["lev_today"] = lev; F["inv_today"] = inv
    # ── MOM_LONG: LONG_HEAT 3-leg (DECISION_LOG 208) then LONG_CHOP_BURST (201), sequential ──
    ml = F.S == "MOM_LONG"
    hot = (ml & (pd.to_numeric(F.entry_btc_ema20_slope, errors="coerce") >= 0.07) & (pd.to_numeric(F.entry_btc_rsi_prev, errors="coerce") >= 64)
           & (pd.to_numeric(F.entry_bull_pct, errors="coerce") >= 80) & (pd.to_numeric(F.entry_btc_off30d_high_pct, errors="coerce") > -10))
    F.loc[hot & F.keep, "why"] = "LONG_HEAT_BLOCK(3-leg)"; F.loc[hot, "keep"] = False
    F["chop"] = False
    live = F.keep & (F.S != "SURGE_SHORT")
    for sd in sorted(F.seed.unique()):
        ix = F.index[(F.seed == sd) & live]
        ix = ix[np.argsort(F.loc[ix, "tms"].values, kind="stable")]
        kept_t = []
        eff = pd.to_numeric(F.entry_btc_eff72, errors="coerce")
        for i in ix:
            t = F.at[i, "tms"]
            if F.at[i, "S"] == "MOM_LONG" and eff[i] <= 0.007:
                j = np.searchsorted(kept_t, t - 120_000, side="left")
                if j < len(kept_t) and kept_t[j] <= t:
                    F.at[i, "keep"] = False; F.at[i, "why"] = "LONG_CHOP_BURST"; F.at[i, "chop"] = True; F.at[i, "usd"] = 0.0
                    continue
            kept_t.append(t)
    F["usd"] = np.where(F.keep, F.usd, 0.0)
    return F


# ───────────────────────────────────────────── add-on cohorts ─────────────────────────────────────────────
def heat_readmits(ml_inv_mean):
    """re-scope blocks (bull ≥ 85) that today's 3-leg rule admits again — priced by HEAT_REGIME_REVIEW with the live ML exit replica."""
    H = pd.read_csv(os.path.join(REP, "HEAT_YR5_SIGNALS_priced.csv"))
    R = BT.btc_at(H.t.values, "forming")
    o30 = pd.read_csv(os.path.join(REP, "backtest_cache", "btc_off30d_5m.csv")).sort_values("ts_ms")
    j = np.searchsorted(o30.ts_ms.values, H.t.values, side="right") - 1
    H["off30"] = np.where(j >= 0, o30.btc_off30d_high_pct.values[np.maximum(j, 0)], np.nan)
    H["slope"], H["rsi_prev"] = R.slope.values, R.rsi_prev.values
    still = (H.slope >= 0.07) & (H.rsi_prev >= 64) & (H.off30 > -10)
    A = H[~still].copy()
    A["w"] = A.n_seeds / 3.0
    A["t"] = pd.to_datetime(A.t, unit="ms")
    A["usd"] = BOOK * SLOT * ml_inv_mean * 20 * A.pct / 100
    return A, int(still.sum()), len(H)


def lite():
    K = pd.read_pickle(os.path.join(SCR, "lite_streak", "kept.pkl"))["K"][12].copy()
    R = BT.btc_at(K.entry_ts.values, "closed")
    K["bear"] = [BT.bearish(a, b) is True for a, b in zip(R.ret1d, R.gap)]
    armed = (K.why_pri == "lock") | ((K.why_pri == "cap") & (K.PRI >= 1.9))
    K["pct"] = np.where(armed, TP - 0.10, K.PRI)                # fixed +3 net touched → +3 − the ruler's 0.10 exit slip
    K["t"] = pd.to_datetime(K.entry_ts, unit="ms")
    K["keep"] = ~K.bear
    K["usd"] = np.where(K.keep, BOOK * SLOT * LEV(0.32) * K.pct / 100, 0.0)
    return K


def willy():
    W = pd.read_csv(os.path.join(SCR, "yr5h", "willy_priced.csv"))
    W = W[W.st == "ok"].copy()
    W["t"] = pd.to_datetime(W.te, unit="ms")
    W["usd"] = BOOK * SLOT * LEV(1.0) * W.pct / 100
    return W


def stats(rows, nseeds):
    """rows: DataFrame with t, pct, usd, (w). → dict per half."""
    out = {}
    for h, (a, b) in {"H1": (T0, TS), "H2": (TS, T1), "FY": (T0, T1)}.items():
        x = rows[(rows.t >= a) & (rows.t < b)]
        w = x.w.values if "w" in x else np.ones(len(x))
        if len(x) == 0:
            out[h] = dict(N=0, WR=np.nan, avg=np.nan, usd=0.0, lo=np.nan, hi=np.nan, days=0); continue
        lo, hi = boot_day(x.pct.values, x.t.dt.floor("D").values, w)
        out[h] = dict(N=w.sum() / nseeds, WR=100 * (w * (x.pct > 0)).sum() / w.sum(), avg=(w * x.pct).sum() / w.sum(),
                      usd=x.usd.sum() / nseeds, lo=lo, hi=hi, days=x.t.dt.floor("D").nunique())
    return out


def main():
    oct6 = "--oct6" in sys.argv
    F = build_replay(oct6)
    ns = F.seed.nunique()
    res = {}
    if oct6:
        for s in SLEEVES:
            x = F[F.S == s]
            if len(x):
                res[s] = stats(x[["t", "pct", "usd"]], ns)
        for s, v in res.items():
            print(f"{s:<12} FY N {v['FY']['N']:7.1f} WR {v['FY']['WR']:5.1f} avg {v['FY']['avg']:+.3f} $ {v['FY']['usd']:+9.0f}")
        return
    K = F[F.keep]
    for s in ["MOM_LONG", "MOM_SHORT", "FLIP_SHORT", "SPIKE_FADE", "BULLRUN", "BEARRUN", "FRENZY_LONG", "FRENZY_WIDE", "SURGE_LONG", "SURGE_SHORT"]:
        res[s] = stats(K[K.S == s].assign(pct=K.pct_today)[["t", "pct", "usd"]], ns)
    ml_inv = K[K.S == "MOM_LONG"].inv_today.mean()
    A, still, nh = heat_readmits(ml_inv)
    res["MOM_LONG_HEAT_READMIT"] = stats(A[["t", "pct", "usd", "w"]], 1)
    ml_plus = pd.concat([K[K.S == "MOM_LONG"].assign(pct=K.pct_today, w=1.0)[["t", "pct", "usd", "w"]],
                         A[["t", "pct", "usd", "w"]].assign(usd=A.usd * ns, w=A.w * ns)])   # per-seed weights → pooled-seed units
    res["MOM_LONG_INCL_READMIT"] = stats(ml_plus, ns)
    L = lite(); res["FRENZY_LITE"] = stats(L[L.keep][["t", "pct", "usd"]], 1)
    W = willy(); res["FRENZY_WILLY"] = stats(W[["t", "pct", "usd"]], 1)
    for k in ("A", "B"):
        res[f"FRENZY_WILLY_{k}"] = stats(W[W.trigger == k][["t", "pct", "usd"]], 1)
    # cross-check: vol/mcap study's FRENZY TODAY tick cohort (FIX +3/−3 at 8 s, engine-parity entries, bearish block on)
    V = pd.read_csv(os.path.join(REP, "FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv"))
    V = V[V.cohort == "frenzy_today_bear"].copy(); V["t"] = pd.to_datetime(V.t0, unit="ms")
    V["usd"] = 0.0
    for s, lv in (("FRENZY", None), ("WIDE", LEV(0.2))):
        x = V[V.sleeve == s].copy()
        x["usd"] = BOOK * SLOT * (LEV(0.32) if lv is None else lv) * x.pct / 100
        res[f"XCHECK_{s}_TICK_COHORT"] = stats(x[["t", "pct", "usd"]], 1)
    # totals (each strategy on its own flat $3k book → $ add up; avg % = pooled per-trade mean)
    rep_live = K[K.S != "SURGE_SHORT"].assign(pct=K.pct_today)[["t", "pct", "usd"]].assign(w=1.0)
    adds = [A[["t", "pct", "usd", "w"]].assign(usd=A.usd * ns, w=A.w * ns)]
    lit = L[L.keep][["t", "pct", "usd"]].assign(usd=lambda d: d.usd * ns, w=float(ns))
    wil = W[["t", "pct", "usd"]].assign(usd=lambda d: d.usd * ns, w=float(ns))
    res["TOTAL_REPLAY_ONLY"] = stats(pd.concat([rep_live] + adds), ns)
    res["TOTAL_LIVE_NO_WILLY"] = stats(pd.concat([rep_live] + adds + [lit]), ns)
    res["TOTAL_LIVE_WITH_WILLY"] = stats(pd.concat([rep_live] + adds + [lit, wil]), ns)
    # per-seed FY $ (replay sleeves)
    seeds = K.groupby(["S", "seed"]).usd.sum().unstack()
    # ── diagnostics ──
    diag = {}
    diag["refusals_per_seed"] = {f"{a}|{b}": v for (a, b), v in (F[~F.keep].groupby(["S", "why"]).size() / ns).round(1).items()}
    diag["moved_wide_to_long_per_seed"] = round(float((F.note != "").sum() / ns), 1)
    w = F[F.sleeve == "FRENZY_WIDE"]
    diag["wide_streak_join"] = f"{w.above_streak.notna().mean() * 100:.1f}% joined; ATR agrees on {w.streak_atr_match[w.above_streak.notna()].mean() * 100:.1f}% of joined"
    fz = F[F.S.isin(["FRENZY_LONG", "FRENZY_WIDE"]) & F.keep]
    diag["frenzy_tp_reprice"] = f"{(fz.pk >= TP).mean() * 100:.1f}% of kept FRENZY/WIDE fills peaked ≥ +3 → booked +3 (were {fz[fz.pk >= TP].pct.mean():+.2f} avg under the +4 TP)"
    diag["heat_readmit"] = f"{nh} re-scope-blocked signals; today's 3-leg rule still blocks {still}; re-admitted {nh - still} ({A.w.sum():.1f}/seed)"
    diag["lite"] = f"{len(L)} study fills; bearish-day blocks {int((~L.keep).sum())}; TP re-price armed {int(((L.why_pri == 'lock') | ((L.why_pri == 'cap') & (L.PRI >= 1.9))).sum())}"
    diag["fade_ticket"] = f"median ticket ratio on a $3k book {np.nanmedian(F.fade_ticket_ratio):.2f} (replay fill ratio median {F[F.S == 'SPIKE_FADE'].fr.median():.2f})"
    diag["ml_inv_mean_today"] = round(float(ml_inv), 3)
    diag["size_factor_by_sleeve"] = F[F.S.isin(["MOM_LONG", "MOM_SHORT", "FLIP_SHORT"])].groupby("S").size_factor.describe()[["mean", "min", "max"]].round(3).to_dict("index")
    json.dump({"res": res, "diag": diag, "seeds_fy_usd": seeds.round(0).to_dict()}, open(os.path.join(SCR, "yr5h", "res.json"), "w"), indent=1, default=str)
    F.to_pickle(os.path.join(SCR, "yr5h", "F_today.pkl")); L.to_pickle(os.path.join(SCR, "yr5h", "L.pkl")); A.to_pickle(os.path.join(SCR, "yr5h", "A.pkl"))
    for s, v in res.items():
        print(f"{s:<24}" + " | ".join(f"{h} N {v[h]['N']:6.1f} WR {v[h]['WR']:5.1f} avg {v[h]['avg']:+.3f} [{v[h]['lo']:+.2f},{v[h]['hi']:+.2f}] $ {v[h]['usd']:+8.0f}" for h in ("H1", "H2", "FY")))
    print(json.dumps(diag, indent=1, default=str, ensure_ascii=False))
    print(seeds.round(0).to_string())


if __name__ == "__main__":
    main()
