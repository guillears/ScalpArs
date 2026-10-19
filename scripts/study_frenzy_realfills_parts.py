#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — Parts C (70× early entry), D (LITE and the volume boundary), E (entry timeline),
F (LITE window / green bar). Reads the engine replica outputs (study_frenzy_realfills_engine.py) + the real fills + journals.
Frozen rules are in the engine docstring; this script only tabulates. LODO / day-bootstrap where a choice is made.
Usage: venv/bin/python scripts/study_frenzy_realfills_parts.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_engine as EN  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
RF = RC.RF
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
MS5, MIN, HR = 300_000, 60_000, 3_600_000
NOTL = RC.NOTIONAL
rng = np.random.default_rng(3)


def dayof(ms):
    return pd.to_datetime(ms, unit="ms").strftime("%Y-%m-%d")


def dboot_daily(dsum_a, dsum_b=None, nb=4000):
    """paired day-bootstrap of Σ (a − b) over days: dsum_* = Series indexed by day (missing = 0)."""
    idx = sorted(set(dsum_a.index) | set(dsum_b.index if dsum_b is not None else []))
    a = dsum_a.reindex(idx, fill_value=0).values; b = dsum_b.reindex(idx, fill_value=0).values if dsum_b is not None else 0 * a
    d = a - b
    if not len(d):
        return 0, np.nan, np.nan, np.nan
    bs = d[rng.integers(0, len(d), (nb, len(d)))].sum(1)
    return float(d.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())


def summ(F, label):
    if not len(F):
        return dict(label=label, N=0)
    ds = F.groupby(F.te.map(dayof)).usd.sum()
    return dict(label=label, N=len(F), N_long=int((F.sleeve == "FRENZY_LONG").sum()), N_lite=int((F.sleeve == "FRENZY_LITE").sum()),
                WR=float((F.pct > 0).mean()), avg_pct=float(F.pct.mean()), sum_pct=float(F.pct.sum()), usd=float(F.usd.sum()),
                usd_long=float(F[F.sleeve == "FRENZY_LONG"].usd.sum()), usd_lite=float(F[F.sleeve == "FRENZY_LITE"].usd.sum()),
                worst_day_usd=float(ds.min()), days=len(ds), provisional_1m=int((F.n_bar > 0).sum()))


def lodo_choice(daily, names, base):
    """daily: DataFrame day × variant of Σ$; choose the best variant on other days, apply to the held-out day."""
    oos = []; picks = []
    for d in daily.index:
        tr = daily.drop(index=d)
        j = tr[names].sum().idxmax(); picks.append(j); oos.append(daily.loc[d, j])
    o = pd.Series(oos, index=daily.index)
    tot, lo, hi, p = dboot_daily(o, daily[base])
    return dict(oos_usd=float(o.sum()), today_usd=float(daily[base].sum()), delta=tot, d_lo=lo, d_hi=hi, P=p,
                picks=pd.Series(picks).value_counts().to_dict())


def main():
    SF = pd.read_pickle(f"{RF}/sim_fills.pkl"); SL = pd.read_pickle(f"{RF}/sim_log.pkl"); W = pd.read_pickle(f"{RF}/walk_all.pkl")
    PR = pd.read_pickle(f"{RF}/cands_priced.pkl")
    D = pd.read_pickle(f"{RF}/real.pkl"); meta = pd.read_pickle(f"{RF}/grid_meta.pkl"); V = np.load(f"{RF}/grid_V.npy")
    D["today"] = meta["ref"]; D["usd"] = D.today / 100 * D.sleeve.map(NOTL)
    D["sig_close"] = (D.te - 1000) // MS5 * MS5
    RP = EN.real_positions()
    SF["key"] = SF.pair + "|" + SF.t_close.astype(str)
    base = SF[SF.variant == "T100 (today)"]
    out = {}

    # ───────── parity: real LONG / LITE fills found in the replica at T100 ─────────
    par = []
    for r in D[D.sleeve.isin(["FRENZY_LONG", "FRENZY_LITE"])].itertuples():
        m = base[(base.pair == r.pair) & (base.t_close == r.sig_close)]
        lg = SL[(SL.variant == "T100 (today)") & (SL.pair == r.pair) & (SL.t_close == r.sig_close)]
        wrow = W[(W["T"] == 100) & (W.pair == r.pair) & (W.t_close == r.sig_close)]
        par.append(dict(key=r.key, sleeve=r.sleeve, era=r.era, found_same_sleeve=bool(len(m) and (m.sleeve == r.sleeve).any()),
                        found_other=bool(len(m) and not (m.sleeve == r.sleeve).any()),
                        log_code=";".join(lg.code.astype(str)) if len(lg) else "", walk_bar=bool(len(wrow)),
                        walk_fresh_on=bool(len(wrow) and wrow.fresh_on.iloc[0]), walk_code=wrow.long_code.iloc[0] if len(wrow) else "",
                        walk_vol=float(wrow.vol_mult.iloc[0]) if len(wrow) else np.nan, stamp_vol=r.entry_frenzy_vol_mult,
                        sim_pct=float(m.pct.iloc[0]) if len(m) else np.nan, real_walk_pct=r.today))
    PAR = pd.DataFrame(par); PAR.to_csv(f"{OUT}_D_parity.csv", index=False)
    out["parity"] = PAR.groupby("sleeve")[["found_same_sleeve", "walk_bar"]].mean()

    # ───────── PART D1: simulated before / after per variant ─────────
    rows = []
    for v in SF.variant.unique():
        F = SF[SF.variant == v]
        rows.append(summ(F, v))
        for e in sorted(F.era.unique()):
            x = summ(F[F.era == e], v); x["era"] = e; rows.append(x)
    S1 = pd.DataFrame(rows); S1.to_csv(f"{OUT}_D_sim_variants.csv", index=False)
    daily = SF.assign(day=SF.te.map(dayof)).pivot_table(index="day", columns="variant", values="usd", aggfunc="sum", fill_value=0)
    for v in daily.columns:
        if v != "T100 (today)":
            t, lo, hi, p = dboot_daily(daily[v], daily["T100 (today)"])
            S1.loc[(S1.label == v) & S1.era.isna(), ["d_usd", "d_lo", "d_hi", "P_gt0"]] = [t, lo, hi, p]
    S1.to_csv(f"{OUT}_D_sim_variants.csv", index=False)
    out["lodo_T"] = lodo_choice(daily, ["T100 (today)", "T85", "T70", "T50"], "T100 (today)")

    # ───────── PART D2: real-fill hybrid (owner change / gone / new) ─────────
    hyb = []
    base_keys = set(base.key)
    for v in ("T85", "T70", "T50"):
        F = SF[SF.variant == v]; fk = F.set_index("key")
        rows_v = []
        for r in D[D.sleeve.isin(["FRENZY_LONG", "FRENZY_LITE"])].itertuples():
            k = f"{r.pair}|{r.sig_close}"
            in_base = k in base_keys
            if not in_base:
                st, new_sleeve, new_usd, new_pct = "UNMATCHED (kept as traded)", r.sleeve, r.usd, r.today
            elif k in fk.index:
                x = fk.loc[k]; x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
                if x.sleeve == r.sleeve:
                    st, new_sleeve, new_usd, new_pct = "SAME", r.sleeve, r.usd, r.today
                else:
                    st, new_sleeve, new_usd, new_pct = f"OWNER {r.sleeve[7:]}→{x.sleeve[7:]}", x.sleeve, float(x.usd), float(x.pct)
            else:
                st, new_sleeve, new_usd, new_pct = "GONE", "", 0.0, np.nan
            rows_v.append(dict(variant=v, key=r.key, pair=r.pair, era=r.era, day=r.day, sleeve=r.sleeve, vol_stamp=r.entry_frenzy_vol_mult,
                               status=st, before_pct=r.today, before_usd=r.usd, after_sleeve=new_sleeve, after_pct=new_pct, after_usd=new_usd))
        realk = {f"{r.pair}|{r.sig_close}" for r in D.itertuples()}
        for x in F[~F.key.isin(base_keys) & ~F.key.isin(realk)].itertuples():
            rows_v.append(dict(variant=v, key=x.key, pair=x.pair, era=x.era, day=dayof(x.te), sleeve="", vol_stamp=x.vol_mult, status="NEW " + x.sleeve[7:],
                               before_pct=np.nan, before_usd=0.0, after_sleeve=x.sleeve, after_pct=x.pct, after_usd=x.usd))
        hyb += rows_v
    HY = pd.DataFrame(hyb); HY.to_csv(f"{OUT}_D_hybrid_rows.csv", index=False)
    hs = []
    for v in ("T85", "T70", "T50"):
        H = HY[HY.variant == v]
        for e in ["ALL"] + sorted(H.era.dropna().unique()):
            h = H if e == "ALL" else H[H.era == e]
            hs.append(dict(variant=v, era=e, n_same=int((h.status == "SAME").sum()), n_owner=int(h.status.str.startswith("OWNER").sum()),
                           n_gone=int((h.status == "GONE").sum()), n_new=int(h.status.str.startswith("NEW").sum()),
                           n_unmatched=int(h.status.str.startswith("UNMATCHED").sum()),
                           before_usd=float(h.before_usd.sum()), after_usd=float(h.after_usd.sum()), delta_usd=float(h.after_usd.sum() - h.before_usd.sum())))
        bd = H.groupby("day").before_usd.sum(); ad = H.groupby("day").after_usd.sum()
        t, lo, hi, p = dboot_daily(ad, bd)
        hs[-len(H.era.dropna().unique()) - 1].update(d_lo=lo, d_hi=hi, P_gt0=p, worst_day_before=float(bd.min()), worst_day_after=float(ad.min()))
    pd.DataFrame(hs).to_csv(f"{OUT}_D_hybrid_summary.csv", index=False)

    # ───────── PART D3: LITE entry volume multiple as a separator (real fills; replica; backtest footnote) ─────────
    bands = [(0, 20), (20, 50), (50, 70), (70, 100)]
    LT = D[D.sleeve == "FRENZY_LITE"].copy(); LT["vm"] = pd.to_numeric(LT.entry_frenzy_vol_mult, errors="coerce")
    rb = []
    for nm, X, col in (("real", LT, "vm"), ("replica T100", base[base.sleeve == "FRENZY_LITE"].assign(today=lambda x: x.pct, day=lambda x: x.te.map(dayof)), "vol_mult")):
        for a, b in bands:
            z = X[(X[col] >= a) & (X[col] < b)]
            rb.append(dict(block=nm, band=f"{a}-{b}x", N=len(z), days=z.day.nunique(), WR=float((z.today > 0).mean()) if len(z) else np.nan,
                           avg=float(z.today.mean()) if len(z) else np.nan, usd=float(z.today.sum() / 100 * NOTL["FRENZY_LITE"]),
                           tp2_share=float((z.get("tp", pd.Series(np.nan, index=z.index)) == 2).mean()) if "tp" in z else np.nan))
    # backtest footnote (yr6 seed 1, < 10-03, today's LITE C walked)
    Mb = np.load(f"{RF}/bt_grid.npy"); Bb = pd.read_pickle(f"{RF}/bt_grid_rows.pkl"); coh = pd.read_pickle(f"{RC.SCR}/rid/cohort.pkl")
    coh = coh[coh.blk == "BT"].reset_index(drop=True)
    refb = Mb[:, len(EG.cells())]
    bl = coh.sleeve.values == "FRENZY_LITE"
    for a, b in bands:
        m = bl & (coh.entry_frenzy_vol_mult.values >= a) & (coh.entry_frenzy_vol_mult.values < b) & np.isfinite(refb)
        rb.append(dict(block="backtest yr6 s1", band=f"{a}-{b}x", N=int(m.sum()), days=int(pd.Series(Bb.te.values[m]).map(dayof).nunique()),
                       WR=float((refb[m] > 0).mean()), avg=float(refb[m].mean()), usd=float(refb[m].sum() / 100 * NOTL["FRENZY_LITE"])))
    pd.DataFrame(rb).to_csv(f"{OUT}_D_lite_volbands.csv", index=False)
    # LODO floor choice on real LITE fills + day-shuffle null of the best in-sample floor gain
    floors = [0, 10, 15, 20, 30, 50]
    def floor_gain(x, vm, days):
        g = {f: float(x[(vm >= f)].sum()) for f in floors}
        return g
    x, vm, dd = LT.today.values, LT.vm.values, LT.day.values
    ins = floor_gain(x, vm, dd); best = max(ins, key=ins.get)
    oos = np.zeros(len(x)); pk = []
    for d in np.unique(dd):
        tr = dd != d
        g = {f: float(x[tr & (vm >= f)].sum()) for f in floors}; f = max(g, key=g.get); pk.append(f)
        te_ = ~tr
        oos[te_] = np.where(vm[te_] >= f, x[te_], 0.0)
    nul = []
    for _ in range(2000):
        perm = rng.permutation(np.unique(dd)); xs = np.concatenate([x[dd == d] for d in perm])[:len(x)]
        g = {f: float(xs[(vm >= f)].sum()) for f in floors}; nul.append(max(g.values()) - g[0])
    LF = dict(insample=ins, best=best, lodo_oos_sum_pct=float(oos.sum()), today_sum_pct=float(x.sum()), lodo_picks=pd.Series(pk).value_counts().to_dict(),
              null_p_best_gain=float(np.mean(np.array(nul) >= ins[best] - ins[0])))
    # expectancy bar on the < 20x band (real)
    z = LT[LT.vm < 20]; wins = LT[LT.today > 0].today; losses = LT[LT.today <= 0].today
    be = abs(losses.mean()) / (wins.mean() + abs(losses.mean()))
    zb = [z.today[z.day == d].values for d in z.day.unique()]
    mu, lo, hi, p = RC.day_boot(zb)
    LF.update(band_lt20=dict(N=len(z), days=z.day.nunique(), WR=float((z.today > 0).mean()), breakeven_WR=float(be), avg=mu, ci=(lo, hi),
                             P_avg_lt0=1 - p, top_pair_loss_share=float(z[z.today < 0].groupby("pair").today.sum().min() / z[z.today < 0].today.sum())
                             if (z.today < 0).any() else np.nan))
    pd.Series(LF, dtype=object).to_json(f"{OUT}_D_lite_floor.json", default_handler=str, indent=1)
    out["lite_floor"] = LF

    # ───────── PART C: 70× early entry ─────────
    W70 = W[(W["T"] == 70) & W.fresh_on & W.long_ready]
    W100 = W[(W["T"] == 100) & W.fresh_on & W.long_ready]
    crow = []
    for T_ in (50, 70, 85):
        WT_ = W[(W["T"] == T_) & W.fresh_on & W.long_ready]
        for r in D[D.sleeve.isin(["FRENZY_LONG", "FRENZY_LITE"])].itertuples():
            prev100 = W100[(W100.pair == r.pair) & (W100.t_close < r.sig_close) & (W100.t_close > r.sig_close - 96 * HR)].t_close
            lo_t = max(prev100.max() if len(prev100) else 0, r.sig_close - 96 * HR)
            e = WT_[(WT_.pair == r.pair) & (WT_.t_close > lo_t) & (WT_.t_close < r.sig_close)]
            if not len(e):
                crow.append(dict(T=T_, key=r.key, sleeve=r.sleeve, pair=r.pair, day=r.day, has_earlier=False)); continue
            e = e.sort_values("t_close").iloc[0]
            pr = PR.get(f"{r.pair}|{int(e.t_close)}", {})
            te70 = int(e.t_close) + 8000; E70 = pr.get("E", np.nan)
            # path from the earlier entry to the real fill time: MAE before the real fill, stop hits before it
            mae, hit3 = np.nan, False
            try:
                P = RC.gross_path(r.pair, te70, float(E70), 720 + int((r.te - te70) / MIN))
                k = int(np.searchsorted(P["t"], r.te))
                if k:
                    mae = float(P["lo"][:k].min()); hit3 = bool(mae <= -3)
            except Exception:  # noqa: BLE001
                pass
            crow.append(dict(T=T_, key=r.key, sleeve=r.sleeve, pair=r.pair, day=r.day, has_earlier=True, t_early=pd.Timestamp(e.t_close, unit="ms"),
                             gap_h=(r.sig_close - e.t_close) / HR, price_gap_pct=(E70 / r.E - 1) * 100 if np.isfinite(E70) else np.nan,
                             vol_early=e.vol_mult, hours_early=e.hours, mae_before_real_fill=mae, early_hit_m3_before_real=hit3,
                             early_today=pr.get("long_pct", np.nan), early_why=pr.get("long_why", ""), real_today=r.today,
                             early_scale=pr.get("scale", {}), early_mfe=pr.get("mfe", np.nan), disloc=pr.get("disloc", np.nan)))
    CR = pd.DataFrame(crow); CR.drop(columns=["early_scale"], errors="ignore").to_csv(f"{OUT}_C_real_fill_pairs.csv", index=False)
    # replica-wide: first T-READY bar of every T episode, classed vs 100×
    cls = []
    for T_ in (50, 70, 85):
        WT_ = W[(W["T"] == T_) & W.fresh_on & W.long_ready].sort_values("t_close")
        first = WT_.groupby(["pair", "spike_ts"]).head(1)
        for e in first.itertuples():
            later = W100[(W100.pair == e.pair) & (W100.t_close >= e.t_close) & (W100.t_close <= e.spike_ts + 96 * HR)]
            if len(later) and later.t_close.min() == e.t_close:
                c = "SAME"
            elif len(later):
                c = "LATER"
            else:
                c = "NEVER"
            pr = PR.get(f"{e.pair}|{int(e.t_close)}", {})
            p100 = PR.get(f"{e.pair}|{int(later.t_close.min())}", {}) if len(later) else {}
            row = dict(T=T_, pair=e.pair, t_close=int(e.t_close), day=dayof(e.t_close), era=e.era, cls=c, hours=e.hours, vol=e.vol_mult,
                       early_today=pr.get("long_pct", np.nan), early_mfe=pr.get("mfe", np.nan), early_run5=pr.get("run5", np.nan),
                       t_hit_m3=pr.get("t_hit_m3", np.nan), t_hit_m5=pr.get("t_hit_m5", np.nan), t_hit_m8=pr.get("t_hit_m8", np.nan),
                       gap_h=((later.t_close.min() - e.t_close) / HR) if len(later) else np.nan,
                       price_gap_pct=((pr.get("E", np.nan) / p100.get("E", np.nan) - 1) * 100) if p100 else np.nan,
                       t100_today=p100.get("long_pct", np.nan) if p100 else np.nan, disloc=pr.get("disloc", np.nan))
            for k_, v_ in (pr.get("scale", {}) or {}).items():
                row["st_" + k_] = v_
            cls.append(row)
    CC = pd.DataFrame(cls); CC.to_csv(f"{OUT}_C_replica_first_bars.csv", index=False)
    # scale-in starter: LODO choice of (stop, TP) over starter cells, on 70× first bars that are NOT the same bar as 100×
    st_cols = [c for c in CC.columns if c.startswith("st_")]
    S70 = CC[(CC["T"] == 70) & (CC.cls != "SAME") & (CC.disloc == False)].copy()  # noqa: E712
    sc = dict(N=len(S70), days=S70.day.nunique(), by_cell={c: float(S70[c].mean()) for c in st_cols})
    oos = []; pk = []
    for d in S70.day.unique():
        tr = S70[S70.day != d]; c = tr[st_cols].mean().idxmax(); pk.append(c)
        oos += list(S70.loc[S70.day == d, c].values)
    sc.update(lodo_oos_avg=float(np.mean(oos)) if oos else np.nan, lodo_picks=pd.Series(pk).value_counts().to_dict(),
              today_style_avg=float(S70["st_s3.0_t1.5"].mean()) if len(S70) else np.nan)
    dsum = S70.groupby("day")[st_cols].sum()
    sc["worst_day_pct_by_cell"] = dsum.min().to_dict()
    pd.Series(sc, dtype=object).to_json(f"{OUT}_C_scalein.json", default_handler=str, indent=1)
    out["scalein"] = sc

    # ───────── PART E: entry timeline ─────────
    J = pd.read_pickle(f"{RC.SCR}/journal.pkl")
    Jb = J[(J.e == "BLOCK") & J.gate.astype(str).str.contains("FRENZY")].copy(); Jb["tm"] = pd.to_datetime(Jb.t.astype(str).str[:19])
    allreal = RP.copy(); allreal["sig_close"] = (allreal.t_open - 1000) // MS5 * MS5
    w100 = W[W["T"] == 100].set_index(["pair", "t_close"])
    eps = []
    for r in allreal.itertuples():
        try:
            sp = int(w100.loc[(r.pair, r.sig_close)].spike_ts)
        except Exception:  # noqa: BLE001
            sp = int(pd.Timestamp(str(r.opened_at)[:19]).value // 1e6 - float(r.entry_frenzy_hours or 0) * HR) if pd.notna(r.entry_frenzy_hours) else None
        eps.append(sp)
    allreal["spike_ts"] = eps
    allreal = allreal.merge(D[["key", "today"]].assign(opened_at=D.key.str.split("|").str[0]), how="left",
                            left_on=allreal.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + allreal.pair + "|LONG",
                            right_on="key")
    allreal["mfe"] = np.nan
    PP = pd.read_pickle(f"{RF}/real_paths.pkl")
    for i, r in allreal.iterrows():
        if isinstance(r.key, str) and r.key in PP:
            allreal.loc[i, "mfe"] = RC.mfe_before(PP[r.key])[0]
    allreal["h_since_spike"] = (allreal.t_open - allreal.spike_ts) / HR
    tl = []
    for (pair, sp), g in allreal.groupby(["pair", "spike_ts"]):
        ew = W[(W.pair == pair) & (W.spike_ts == sp)]
        def first_on(T_, ready=False):
            x = ew[(ew["T"] == T_) & ew.fresh_on & (ew.long_ready if ready else True)]
            return round((x.t_close.min() - sp) / HR, 1) if len(x) else None
        lon = ew[(ew["T"] == 100) & ew.fresh_on].sort_values("t_close")
        lite_j = SL[(SL.variant == "T100 (today)") & (SL.pair == pair) & (SL.sleeve == "FRENZY_LITE") & (SL.spike_ts == sp)]
        jb = Jb[(Jb.pair == pair) & (Jb.tm >= pd.Timestamp(sp, unit="ms")) & (Jb.tm <= pd.Timestamp(sp + 96 * HR, unit="ms"))]
        tl.append(dict(pair=pair, spike=pd.Timestamp(sp, unit="ms"),
                       fills="; ".join(f"{s[7:]}@{h:.1f}h {t:+.2f}% (mfe {m:.1f})" if pd.notna(t) else f"{s[7:]}@{h:.1f}h"
                                       for s, h, t, m in zip(g.entry_strategy, g.h_since_spike, g.today, g.mfe)),
                       long_on_bars="; ".join(f"{(x.t_close - sp) / HR:.1f}h {x.vol_mult:.0f}x {x.long_code.replace('FRENZY_', '')}" for x in lon.itertuples()),
                       lite_judged="; ".join(f"{x.hours:.1f}h {x.code}" for x in lite_j.itertuples() if x.code not in ("WILLY_HOLD",)),
                       first_on_50=first_on(50), first_on_70=first_on(70), first_on_85=first_on(85), first_on_100=first_on(100),
                       first_ready_70=first_on(70, True), first_ready_100=first_on(100, True),
                       journal_blocks="; ".join(f"{k.replace('FRENZY_', '')}×{v}" for k, v in jb.gate.value_counts().items())))
    TL = pd.DataFrame(tl).sort_values("spike"); TL.to_csv(f"{OUT}_E_timeline.csv", index=False)
    bk = [(0, 2), (2, 6), (6, 16.75), (16.75, 96.1)]
    ag = []
    for nm, X, hcol, pcol, rcol in (("real", allreal[allreal.today.notna()], "h_since_spike", "today", "mfe"),
                                    ("replica T100", base, "hours", "pct", "mfe")):
        for a, b in bk:
            for s in ("FRENZY_WILLY", "FRENZY_LITE", "FRENZY_LONG", "ALL"):
                sl_ = X.entry_strategy if "entry_strategy" in X else X.sleeve
                z = X[(X[hcol] >= a) & (X[hcol] < b) & ((sl_ == s) if s != "ALL" else True)]
                if len(z):
                    ag.append(dict(block=nm, bucket=f"{a}-{b}h", sleeve=s, N=len(z), WR=float((z[pcol] > 0).mean()), avg=float(z[pcol].mean()),
                                   run5=float((z[rcol] >= 5).mean())))
    for a, b in bk:
        for c in ("SAME", "LATER", "NEVER"):
            z = CC[(CC["T"] == 70) & (CC.cls == c) & (CC.hours >= a) & (CC.hours < b)]
            if len(z):
                ag.append(dict(block="replica 70x first bars", bucket=f"{a}-{b}h", sleeve=c, N=len(z), WR=float((z.early_today > 0).mean()),
                               avg=float(z.early_today.mean()), run5=float((z.early_mfe >= 5).mean())))
    pd.DataFrame(ag).to_csv(f"{OUT}_E_buckets.csv", index=False)

    # ───────── PART F: LITE window / green bar / late LONG at LITE size ─────────
    fr = []
    for v in [x for x in SF.variant.unique() if "LITE" in x or x == "T100 (today)"]:
        F = SF[SF.variant == v]; fk = set(F.key); bk_ = set(base.key)
        add = F[~F.key.isin(bk_)]; lost = base[~base.key.isin(fk)]
        bd = base.groupby(base.te.map(dayof)).usd.sum(); vd = F.groupby(F.te.map(dayof)).usd.sum()
        t, lo, hi, p = dboot_daily(vd, bd)
        fr.append(dict(variant=v, N=len(F), lite_N=int((F.sleeve == "FRENZY_LITE").sum()), added=len(add), added_lite=int((add.sleeve == "FRENZY_LITE").sum()),
                       added_avg=float(add.pct.mean()) if len(add) else np.nan, added_usd=float(add.usd.sum()), lost=len(lost),
                       lost_long=int((lost.sleeve == "FRENZY_LONG").sum()), lost_usd=float(lost.usd.sum()), usd=float(F.usd.sum()),
                       d_usd=t, d_lo=lo, d_hi=hi, P_gt0=p, worst_day=float(vd.min()) if len(vd) else np.nan,
                       added_hours_med=float(add.hours.median()) if len(add) else np.nan))
    FR = pd.DataFrame(fr); FR.to_csv(f"{OUT}_F_lite_variants.csv", index=False)
    wv = ["T100 (today)"] + [f"T100 LITE window {h} h" for h in (24, 36, 48, 72, 96)]
    out["lodo_hmax"] = lodo_choice(daily, [c for c in wv if c in daily.columns], "T100 (today)")
    # green-bar consumption count (replica T100) + journal
    lg = SL[(SL.variant == "T100 (today)") & (SL.sleeve == "FRENZY_LITE")]
    jg = Jb[Jb.gate.astype(str) == "FRENZY_LITE_GREEN_BAR"]
    out["green"] = dict(replica_stretches_judged=int(lg.stretch.nunique()), replica_green_consumed=int((lg.code == "FRENZY_LITE_GREEN_BAR").sum()),
                        replica_ready=int((lg.code == "OPEN").sum()), journal_green_rows=len(jg), journal_green_pairs=int(jg.pair.nunique()))
    # late LONG (> 16.75 h): LONG at blocker size vs LITE-style (LITE C TP, −3) at LITE size on the same bars
    late = []
    BTf = EN.btc_frame()
    for nm, X in (("real", D[(D.sleeve == "FRENZY_LONG")]), ("replica T100", base[base.sleeve == "FRENZY_LONG"])):
        for r in X.itertuples():
            h = r.entry_frenzy_hours if nm == "real" else r.hours
            if not (pd.notna(h) and float(h) > 16.75):
                continue
            k = f"{r.pair}|{int(r.sig_close if nm == 'real' else r.t_close)}"
            pr = PR.get(k, {})
            tcl = int(r.sig_close if nm == "real" else r.t_close)
            j = int(np.searchsorted(BTf.index.values, tcl, side="right")) - 1
            slope = BTf.h1slope.iloc[j] if j >= 0 else np.nan
            vm = float(r.entry_frenzy_vol_mult if nm == "real" else r.vol_mult)
            tp = 3 if (vm >= 20 and np.isfinite(slope) and slope > 0) else 2
            if nm == "real":
                P = PP[r.key]; lp = RC.walk(P, tp=tp, sl=3.0, cap=720)[0]; lg_ = r.today
            else:
                lp = pr.get(f"lite{tp}_pct", np.nan); lg_ = r.pct
            late.append(dict(block=nm, pair=r.pair, day=r.day if nm == "real" else dayof(r.te), hours=float(h), long_pct=lg_, long_usd=lg_ / 100 * NOTL["FRENZY_LONG"],
                             lite_tp=tp, lite_pct=lp, lite_usd=lp / 100 * NOTL["FRENZY_LITE"]))
    LTD = pd.DataFrame(late); LTD.to_csv(f"{OUT}_F_late_long_vs_lite.csv", index=False)
    # backtest footnote for late LONG: LONG fills with hours > 16.75, FIX tp2 / tp3 at −3 (cells from the BT grid)
    names = [c[1] for c in EG.cells()]
    bl = (coh.sleeve.values == "FRENZY_LONG") & (pd.to_numeric(coh.entry_frenzy_hours, errors="coerce").values > 16.75) & np.isfinite(refb)
    out["late_bt"] = dict(N=int(bl.sum()), long_today=float(refb[bl].mean()), tp2=float(np.nanmean(Mb[bl, names.index("FIX tp2/sl3/720m")])),
                          tp3=float(np.nanmean(Mb[bl, names.index("FIX tp3/sl3/720m")])))
    pd.Series({k: v for k, v in out.items() if k != "parity"}, dtype=object).to_json(f"{OUT}_CDEF_summary.json", default_handler=str, indent=1)
    print(out["parity"]); print({k: v for k, v in out.items() if k != "parity"})


if __name__ == "__main__":
    main()
