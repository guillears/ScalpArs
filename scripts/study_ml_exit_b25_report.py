#!/usr/bin/env python3
"""ML EXIT DEEP study (2026-10-26) — analysis + report. Reads the per-fill engine outputs of study_ml_exit_b25_run.py and writes
reports/ML_EXIT_DEEP_2026-10-26_tables.md (all numeric sections) + CSVs:
  _postexit.csv (per fill post-exit / MAE / MFE), _variant_summary.csv (one row per variant × scope), _variant_by_group.csv,
  _state_summary.csv, _yr5_summary.csv.
The narrative report reports/ML_EXIT_DEEP_2026-10-26.md is written by hand on top of these tables.

PRE-DECLARED VERDICT BAR (fixed 2026-10-26 before the variant grid was read):
  Cohort = KEEP master (today's stack) — the primary block; B25 provisional and PROBE are separate lines.
  Δ = variant − BASE, both through the validated replica on the same path (paired per fill); 'after' = recorded stack_pct + Δ.
  SHIP-GRADE (decided by the gate): Δ > 0 with the BONFERRONI day-block bootstrap CI (α = 0.05 / K, K = every read test incl. the
    state-conditional ones) above 0 · Δ > 0 in ≥ 5 of the era groups with ≥ 3 fills, including G5 B12-B16 and G6 B17-B24 ·
    Δ > 0 without the washed-out window · top day and top pair each < 50 % of the net Δ · stop-type rules read on the live-stopped
    cohort (saved vs deeper) · then the 30–50 % haircut, one change at a time, and a pre-committed revert gate.
  OPERATOR CALL: nominal 95 % CI above 0 but one of the legs above fails.
  KEEP OBSERVING: Δ > 0 point, 95 % CI includes 0, recent groups not negative → frozen observe line.
  REJECTED (decided by the gate): Δ ≤ 0, or the 95 % CI entirely below 0, or the recent groups negative with the rest flat.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_exit_b25_data as DD  # noqa: E402
import study_ml_exit_b25_run as RN  # noqa: E402

RP = DD.RP
OUT = RN.OUT
SEED = 20261026
STOPLIKE = ("STOP", "RH_HARD_STOP", "RH_PREMISE_EXIT", "RH_TIME_EXIT", "BE_LOCK")
READ = [v for v in RN.VARIANTS if not v.startswith("_") and v != "BASE"] + list(RN.PARTIALS)
REF_NOTIONAL = None   # set from the B25 ML median notional (for a $ reading of % at today's ticket)


def q(x, p):
    x = pd.Series(x).dropna()
    return float(np.percentile(x, p)) if len(x) else np.nan


def dist(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return "—"
    return f"{x.median():+.2f} / {q(x, 75):+.2f} / {q(x, 90):+.2f} (mean {x.mean():+.2f})"


def nwr(x):
    x = pd.Series(x).dropna()
    return f"{len(x)}·{(x > 0).mean() * 100:.0f}%·{x.mean():+.3f}" if len(x) else "0"


def boot(vals, days, alpha=0.05, reps=20000, seed=SEED):
    vals = np.asarray(vals, float); days = np.asarray(days)
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=vals, minlength=len(ud)); c = np.bincount(inv, minlength=len(ud))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ud), (reps, len(ud)))
    m = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2))), float((m <= 0).mean())


def conc(x):
    if pd.isna(x):
        return "n/a (net ≤ 0)"
    return f"{x * 100:.0f}%" if x <= 1 else f"{x * 100:.0f}% (one day/pair > the whole net)"


def base_reason(r):
    return RP._base_reason(r) if isinstance(r, str) else "NO_DATA"


def load():
    C = DD.cohort().drop(columns=["src"]).set_index("key")
    V = pd.read_csv(OUT + "_variants.csv")
    D = pd.read_csv(OUT + "_fills_paths.csv").set_index("key")
    P = V.pivot(index="key", columns="variant", values="pct")
    R = V.pivot(index="key", columns="variant", values="reason")
    for p, h in RN.PARTIALS.items():
        P[p] = 0.5 * P["BASE"] + 0.5 * P[h]
        R[p] = R["BASE"].astype(str) + "+" + R[h].astype(str)
    C = C.join(D)
    C["day"] = C.ts.dt.strftime("%Y-%m-%d")
    atr = C.loc[C.block == "MASTER", "atr"]
    lo, hi = np.round(np.percentile(atr, [100 / 3, 200 / 3]), 2)
    C["atr_b"] = np.where(C.atr < lo, "ATR_lo", np.where(C.atr < hi, "ATR_mid", "ATR_hi"))
    C["state_cut"] = f"ATR terciles on KEEP master: lo < {lo} ≤ mid < {hi} ≤ hi"
    return C, P, R, (lo, hi)


def main():
    C, P, R, (alo, ahi) = load()
    L = []; W = L.append
    K = C[C.block.isin(["MASTER", "B25"])]
    M = C[C.block == "MASTER"]
    global REF_NOTIONAL
    REF_NOTIONAL = float(C.loc[C.block == "B25", "notional"].median())

    # ── §A cohort / coverage
    W("## A. Cohort and price-path coverage\n")
    W("| block / era group | fills | tick path | 1m path | no path | post-exit ≥ 60 min covered | WR | avg % (stack, 1×) |")
    W("|---|---|---|---|---|---|---|---|")
    for lab, g in [("KEEP master — all", M)] + [(f"  {e}", M[M.era_group == e]) for e in DD.GROUP_ORDER[:-1]] + \
            [("B25 provisional", C[C.block == "B25"]), ("PROBE (separate)", C[C.block == "PROBE"])]:
        if not len(g):
            continue
        W(f"| {lab} | {len(g)} | {(g.src == 'tick').sum()} | {(g.src == '1m').sum()} | {(~g.src.isin(['tick', '1m', 'mixed'])).sum()} | "
          f"{g.post60.notna().sum()} | {(g.y > 0).mean() * 100:.0f}% | {g.y.mean():+.3f} |")
    W(f"\nMixed tick/1m paths: {(C.src == 'mixed').sum()}. Fills with NO path: {(~C.src.isin(['tick', '1m', 'mixed'])).sum()}.\n")

    # ── §B fidelity
    W("## B. Replica fidelity on this cohort (AS-WAS exit params vs the real exit; bot exits from 2026-06-24 20:20 on)\n")
    C["ASWAS"] = P.get("ASWAS"); C["aswas_r"] = R.get("ASWAS").map(base_reason) if "ASWAS" in R else np.nan
    W("| block | path | N | same exit reason | mean Δ (rep − real) | \\|Δ\\| ≤ 0.05 | corr |")
    W("|---|---|---|---|---|---|---|")
    fx = C[C.ASWAS.notna() & ~C.reason_live.isin(["MANUAL", "TRAIL_OLD"])]
    for (b, s), g in fx.groupby(["block", "src"]):
        W(f"| {b} | {s} | {len(g)} | {(g.aswas_r == g.reason_live).mean() * 100:.0f}% | {(g.ASWAS - g.y_trad).mean():+.3f} | "
          f"{((g.ASWAS - g.y_trad).abs() <= 0.05).mean() * 100:.0f}% | {np.corrcoef(g.ASWAS, g.y_trad)[0, 1]:.3f} |")
    mm = fx[(fx.block != "PROBE") & ((fx.aswas_r != fx.reason_live) | ((fx.ASWAS - fx.y_trad).abs() > 0.15))]
    W("\nKEEP / B25 fills off by > 0.15 or a different reason: " + "; ".join(
        f"{r.pair} {r.era} ({r.src}) live {r.reason_live} {r.y_trad:+.2f} vs replica {r.aswas_r} {r.ASWAS:+.2f}" for r in mm.itertuples()) + "\n")
    W(f"Today's-stack replica (BASE) vs recorded stack_pct on KEEP master: BASE {M.index.map(P['BASE']).to_series().mean():+.3f} vs "
      f"recorded {M.y.mean():+.3f} (old-era fills ran older exits — BASE re-prices them on today's exit).\n")

    # ── §C post-exit
    W("## C. Post-exit run per WINNER (how far price went after our exit) — KEEP master + B25\n")
    W("Units: P&L points at 1× measured from the live exit price ((p − exit) / entry × 100). `max before −x` = the highest point price "
      "reached after our exit before it first pulled back x from its running high (≤ 4 h; censored rows counted at their 4 h high). "
      "`realisable by a −x trail` = that high − x (what a trail of width x after our exit would have banked, before fees/slip). "
      "Cells: median / p75 / p90 (mean).\n")
    Wn = K[(K.y > 0) & K.post_cov.astype(str).eq("True")].copy()
    Wn["giveback"] = Wn.peak_pnl - Wn.y_trad
    W("| exit reason (live) | N | giveback = peak − exit | +5 min | +15 min | +30 min | +60 min | max before −0.3 | max before −0.5 | max before −1.0 | realisable −0.5 trail |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, g in [("ALL winners", Wn)] + [(r, g) for r, g in Wn.groupby("reason_live")]:
        W(f"| {lab} | {len(g)} | {dist(g.giveback)} | {dist(g.post5)} | {dist(g.post15)} | {dist(g.post30)} | {dist(g.post60)} | "
          f"{dist(g['runpb0.3'])} | {dist(g['runpb0.5'])} | {dist(g['runpb1.0'])} | {dist(g['runpb0.5'] - 0.5)} |")
    W("\nShare of winners where price was HIGHER than our exit at +15 / +30 / +60 min: "
      f"{(Wn.post15 > 0).mean() * 100:.0f}% / {(Wn.post30 > 0).mean() * 100:.0f}% / {(Wn.post60 > 0).mean() * 100:.0f}% "
      f"(by ≥ +0.30: {(Wn.post15 >= .3).mean() * 100:.0f}% / {(Wn.post30 >= .3).mean() * 100:.0f}% / {(Wn.post60 >= .3).mean() * 100:.0f}%).\n")
    W("Per era group — money left on the table by winners (Σ over the group's winners, % points at 1×; $ at the B25 ML ticket "
      f"${REF_NOTIONAL:,.0f} notional):\n")
    W("| group | winners | Σ giveback (peak − exit) | Σ max-before-−0.5 | Σ realisable −0.5 trail | median realisable / winner | $ realisable at today's ticket |")
    W("|---|---|---|---|---|---|---|")
    for e in DD.GROUP_ORDER:
        g = Wn[Wn.era_group == e]
        if len(g):
            r = g["runpb0.5"] - 0.5
            W(f"| {e} | {len(g)} | {g.giveback.sum():+.2f} | {g['runpb0.5'].sum():+.2f} | {r.sum():+.2f} | {r.median():+.2f} | ${r.sum() / 100 * REF_NOTIONAL:+,.0f} |")
    # stamp cross-check
    st = K[K.post_cov.astype(str).eq("True")].copy()
    for k in (5, 15, 30):
        s = pd.to_numeric(st[f"post_exit_pnl_at_{k}min"], errors="coerce") - st.y_trad
        ok = s.notna() & st[f"post{k}"].notna()
        if ok.sum() > 5:
            W(f"- live stamp check post_exit_pnl_at_{k}min − exit vs path +{k} min: N {ok.sum()} · corr {np.corrcoef(s[ok], st.loc[ok, f'post{k}'])[0, 1]:.3f} · "
              f"median gap {(st.loc[ok, f'post{k}'] - s[ok]).median():+.3f}")
    Ls = K[(K.y <= 0) & K.post_cov.astype(str).eq("True")]
    W(f"\nLOSERS after the exit (N {len(Ls)}): +15 / +60 min median {Ls.post15.median():+.2f} / {Ls.post60.median():+.2f}; "
      f"share back above ENTRY within 60 min (post-exit max ≥ −exit): {((Ls.postmax60 + Ls.y_trad) > 0).mean() * 100:.0f}%; "
      f"share recovering ≥ +0.40 above entry within 60 min: {((Ls.postmax60 + Ls.y_trad) >= 0.40).mean() * 100:.0f}%.\n")

    # ── §D MAE / MFE
    W("## D. MAE / MFE — winners vs losers (KEEP master + B25, net %, live life of the trade)\n")
    W("| | N | MFE (peak) median / p75 / p90 | MFE minute median | MAE (trough) median / p25 / p10 | MAE minute median |")
    W("|---|---|---|---|---|---|")
    for lab, g in (("winners", K[K.y > 0]), ("losers", K[K.y <= 0])):
        W(f"| {lab} | {len(g)} | {g.mfe.median():+.2f} / {q(g.mfe, 75):+.2f} / {q(g.mfe, 90):+.2f} | {g.mfe_min.median():.1f} | "
          f"{g.mae.median():+.2f} / {q(g.mae, 25):+.2f} / {q(g.mae, 10):+.2f} | {g.mae_min.median():.1f} |")
    Lo = K[K.y <= 0]; Wi = K[K.y > 0]
    W("\nLosers that NEVER went green (peak < x by N minutes):\n")
    W("| threshold | by 5 min | by 10 min | by 15 min |"); W("|---|---|---|---|")
    for x in (0.05, 0.10):
        W(f"| peak < +{x:.2f} | " + " | ".join(f"{(Lo[f'peak_by{n}'] < x).mean() * 100:.0f}% ({(Lo[f'peak_by{n}'] < x).sum()}/{len(Lo)})" for n in (5, 10, 15)) + " |")
    W("\nThe same cut on WINNERS (the cost side of a never-green exit):\n")
    W("| threshold | by 5 min | by 10 min | by 15 min |"); W("|---|---|---|---|")
    for x in (0.05, 0.10):
        W(f"| peak < +{x:.2f} | " + " | ".join(f"{(Wi[f'peak_by{n}'] < x).mean() * 100:.0f}% ({(Wi[f'peak_by{n}'] < x).sum()}/{len(Wi)})" for n in (5, 10, 15)) + " |")
    W("\nWinners that dipped before winning (trough before the first +0.40 touch):\n")
    W("| dip below | share of winners | their avg final % |"); W("|---|---|---|")
    for x in (-0.3, -0.5, -0.7):
        g = Wi[Wi.mae_before_arm < x]
        W(f"| {x:+.1f} | {len(g) / max(len(Wi), 1) * 100:.0f}% ({len(g)}/{len(Wi)}) | {g.y.mean():+.3f} |")
    W(f"\nLive-stamp cross-check: path MFE vs stamped peak_pnl corr {np.corrcoef(K.mfe.fillna(0), K.peak_pnl.fillna(0))[0, 1]:.3f}, median gap "
      f"{(K.mfe - K.peak_pnl).median():+.3f}; MAE vs trough_pnl corr {np.corrcoef(K.mae.fillna(0), K.trough_pnl.fillna(0))[0, 1]:.3f}.\n")
    W("Per era group (winners MFE median · losers never-green share peak < +0.10 by 10 min):\n")
    W("| group | W N | W MFE med | W giveback med | L N | L never-green@10 |"); W("|---|---|---|---|---|---|")
    for e in DD.GROUP_ORDER:
        w, l = Wi[Wi.era_group == e], Lo[Lo.era_group == e]
        W(f"| {e} | {len(w)} | {w.mfe.median():+.2f} | {(w.peak_pnl - w.y_trad).median():+.2f} | {len(l)} | "
          f"{(l.peak_by10 < 0.10).mean() * 100 if len(l) else float('nan'):.0f}% |")

    # ── §E variants
    nstate = len(RN.STATES)
    Kt = len(READ) * (1 + nstate)
    alpha_b = 0.05 / Kt
    W(f"\n## E. Two-sided counterfactuals on today's stack (validated replica) — KEEP master\n")
    W(f"{len(READ)} read variants (partials included) × (whole cohort + {nstate} state-conditional reads) = **K = {Kt} tests**; "
      f"Bonferroni α = 0.05 / {Kt} = {alpha_b:.5f} (two-sided CI {100 * (1 - alpha_b):.3f} %). Day-block bootstrap, 20,000 resamples. "
      "Δ = variant − BASE, paired per fill on the same path. 'Rest' fills = fills whose exit the variant did not touch.\n")
    rows, grows = [], []
    b = P["BASE"]
    for v in READ:
        for scope, g in (("MASTER", M), ("B25", C[C.block == "B25"]), ("PROBE", C[C.block == "PROBE"])):
            idx = g.index
            x, bb = P.loc[idx, v], b.loc[idx]
            d = (x - bb).fillna(0.0)
            if not len(d):
                continue
            lo, hi, p0 = boot(d.values, g.day.values) if g.day.nunique() > 1 else (np.nan, np.nan, np.nan)
            blo, bhi, _ = boot(d.values, g.day.values, alpha=alpha_b) if g.day.nunique() > 1 else (np.nan, np.nan, np.nan)
            bw = bb > 0
            tot = d.sum()
            pos_by_day = d.groupby(g.day.values).sum(); pos_by_pair = d.groupby(g.pair.values).sum()
            top_day = pos_by_day.max() / tot if tot > 0 else np.nan
            top_pair = pos_by_pair.max() / tot if tot > 0 else np.nan
            ls = g.reason_live.isin(STOPLIKE) & R.loc[idx, "BASE"].map(base_reason).isin(STOPLIKE)
            ex_w = ~g.washed.astype(bool)
            row = dict(variant=v, scope=scope, N=len(d), days=g.day.nunique(), base=bb.mean(), after=x.mean(), wr_base=(bb > 0).mean(),
                       wr_after=(x > 0).mean(), d=d.mean(), d_sum=tot, lo95=lo, hi95=hi, p_le0=p0, lo_bonf=blo, hi_bonf=bhi,
                       n_changed=int((d.abs() > 1e-9).sum()), n_up=int((d > 1e-9).sum()), n_down=int((d < -1e-9).sum()),
                       d_on_base_winners=d[bw].sum(), d_on_base_losers=d[~bw].sum(),
                       winners_cut=int(((d < -1e-9) & bw).sum()), losers_worse=int(((d < -1e-9) & ~bw).sum()),
                       losers_saved=int(((d > 1e-9) & ~bw).sum()), winners_more=int(((d > 1e-9) & bw).sum()),
                       top_day_share=top_day, top_pair_share=top_pair, top_day=pos_by_day.idxmax(), top_pair=pos_by_pair.idxmax(),
                       d_ex_washed=d[ex_w.values].mean() if ex_w.any() else np.nan,
                       ls_n=int(ls.sum()), ls_saved=int(((d > 1e-9) & ls).sum()), ls_saved_sum=d[(d > 0) & ls].sum(),
                       ls_deeper=int(((d < -1e-9) & ls).sum()), ls_deeper_sum=d[(d < 0) & ls].sum(),
                       nonls_changed=int(((d.abs() > 1e-9) & ~ls).sum()), nonls_sum=d[~ls].sum(),
                       recorded_before=g.y.mean(), recorded_after=(g.y + d).mean(),
                       usd_at_ticket=tot / 100 * REF_NOTIONAL)
            if scope == "MASTER":
                gp = []
                for e in DD.GROUP_ORDER[:-1]:
                    m_ = (g.era_group == e).values
                    if m_.sum():
                        grows.append(dict(variant=v, group=e, N=int(m_.sum()), base=bb[m_].mean(), after=x[m_].mean(), d=d[m_].mean(),
                                          d_sum=d[m_].sum(), rec_before=g.y[m_].mean(), rec_after=(g.y + d)[m_].mean()))
                        if m_.sum() >= 3:
                            gp.append(d[m_].mean())
                row["groups_pos"] = int(sum(1 for z in gp if z > 1e-9)); row["groups_n"] = len(gp)
                rec = g.era_group.isin(["G5 B12-B16", "G6 B17-B24"]).values
                row["d_recent"] = d[rec].mean(); row["d_G5"] = d[(g.era_group == "G5 B12-B16").values].mean()
                row["d_G6"] = d[(g.era_group == "G6 B17-B24").values].mean()
            rows.append(row)
    S = pd.DataFrame(rows)
    G = pd.DataFrame(grows)

    def verdict(r):
        if not (r.d > 1e-9):
            return "REJECTED (Δ ≤ 0)"
        if r.hi95 < 0:
            return "REJECTED (CI < 0)"
        ship = (r.lo_bonf > 0 and r.groups_pos >= 5 and r.d_G5 > 0 and r.d_G6 > 0 and r.d_ex_washed > 0
                and (r.top_day_share < 0.5) and (r.top_pair_share < 0.5))
        if ship:
            return "SHIP-GRADE"
        if r.lo95 > 0:
            return "OPERATOR CALL"
        if r.d_recent < 0 and r.d_ex_washed <= 0:
            return "REJECTED (recent < 0, ex-washed ≤ 0)"
        return "KEEP OBSERVING"
    SM = S[S.scope == "MASTER"].copy()
    SM["verdict"] = SM.apply(verdict, axis=1)
    S = S.merge(SM[["variant", "verdict"]], on="variant", how="left")
    S.to_csv(OUT + "_variant_summary.csv", index=False)
    G.to_csv(OUT + "_variant_by_group.csv", index=False)

    W("| variant | N·WR·avg BASE → after | Δ %/fill [95 % CI] | Bonf. CI | P(Δ≤0) | groups + | G5 / G6 Δ | ex-washed Δ | Σ Δ on BASE winners / losers | winners cut · losers worse · losers saved · winners more | top day / pair share | $ Σ at ticket | verdict |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in SM.sort_values("d", ascending=False).itertuples():
        lab = r.variant + (" (post-hoc)" if r.variant in RN.POSTHOC else "")
        hc = f" (haircut {r.d * 0.5:+.3f}…{r.d * 0.7:+.3f})" if r.d > 0 else ""
        W(f"| {lab} | {r.N}·{r.wr_base * 100:.0f}%·{r.base:+.3f} → {r.wr_after * 100:.0f}%·{r.after:+.3f} | {r.d:+.3f}{hc} [{r.lo95:+.3f}, {r.hi95:+.3f}] | "
          f"[{r.lo_bonf:+.3f}, {r.hi_bonf:+.3f}] | {r.p_le0:.2f} | {r.groups_pos}/{r.groups_n} | {r.d_G5:+.3f} / {r.d_G6:+.3f} | {r.d_ex_washed:+.3f} | "
          f"{r.d_on_base_winners:+.2f} / {r.d_on_base_losers:+.2f} | {r.winners_cut} · {r.losers_worse} · {r.losers_saved} · {r.winners_more} | "
          f"{conc(r.top_day_share)} / {conc(r.top_pair_share)} | "
          f"${r.usd_at_ticket:+,.0f} | {r.verdict} |")
    W("\nStop-type and early-exit rules on the LIVE-STOPPED cohort (live exit a stop / RH ∧ replica BASE a stop): saved vs deeper; "
      "the rest of the cohort is the cost side (winners/flat fills the rule touches):\n")
    W("| variant | live-stopped N | saved (n · Σ %) | deeper (n · Σ %) | non-stopped fills changed · Σ Δ % |"); W("|---|---|---|---|---|")
    for r in SM[SM.variant.str.startswith("L_") | SM.variant.str.startswith("X_")].itertuples():
        W(f"| {r.variant}{' (post-hoc, wider)' if r.variant in RN.POSTHOC else ''} | {r.ls_n} | {r.ls_saved} · {r.ls_saved_sum:+.2f} | {r.ls_deeper} · {r.ls_deeper_sum:+.2f} | {r.nonls_changed} · {r.nonls_sum:+.2f} |")
    # artefact guard for winner rules: a fill that never armed and was stopped must show Δ = 0
    guard = []
    for v in [v for v in READ if v.startswith("W_") or v.startswith("P_")]:
        na = M[(M.mfe < 0.395) & M.index.map(R["BASE"]).map(base_reason).isin(STOPLIKE)].index
        dd = (P.loc[na, v] - P.loc[na, "BASE"]).abs()
        guard.append(f"{v} {int((dd > 1e-9).sum())}/{len(na)}")
    W("\nArtefact guard — winner-side rules on fills that never reached the +0.40 arm and were stopped (Δ must be 0): "
      + ", ".join(guard) + ".\n")
    W("\nWider-stop guard (post-hoc X_ rules): fills the replica BASE did NOT stop must show Δ = 0 — "
      + ", ".join(f"{v} {int(((P.loc[M.index, v] - P.loc[M.index, 'BASE']).abs() > 1e-9)[~M.index.map(R['BASE']).map(base_reason).isin(STOPLIKE)].sum())} changed"
                  for v in sorted(RN.POSTHOC)) + ".\n")
    W("\nPer era group, before → after (replica BASE avg % → variant avg %, N: before→after) — top variants by Δ:\n")
    top = SM.sort_values("d", ascending=False).variant.head(8).tolist()
    W("| variant | " + " | ".join(DD.GROUP_ORDER[:-1]) + " | B25 | PROBE |"); W("|---|" + "---|" * (len(DD.GROUP_ORDER) + 1))
    for v in top + [v for v in ("L_NG10_0.10", "L_NG15_0.10", "W_ARM0.8", "W_K1.5") if v not in top]:
        cells = []
        for e in DD.GROUP_ORDER[:-1]:
            z = G[(G.variant == v) & (G.group == e)]
            cells.append(f"{int(z.N.iloc[0])}: {z.base.iloc[0]:+.2f}→{z.after.iloc[0]:+.2f}" if len(z) else "—")
        for sc in ("B25", "PROBE"):
            z = S[(S.variant == v) & (S.scope == sc)]
            cells.append(f"{int(z.N.iloc[0])}: {z.base.iloc[0]:+.2f}→{z.after.iloc[0]:+.2f}" if len(z) else "—")
        W(f"| {v} | " + " | ".join(cells) + " |")

    W("\nPer BATCH (era) Σ Δ % points for the top variants (master; B25 separate):\n")
    eras = [e for e in ["BASE"] + [f"B{i}" for i in range(1, 25)] if (M.era == e).any()]
    W("| variant | " + " | ".join(f"{e} ({(M.era == e).sum()})" for e in eras) + " | B25 |"); W("|---|" + "---|" * (len(eras) + 1))
    BT = []
    for v in top[:6]:
        d = (P.loc[M.index, v] - b.loc[M.index]).fillna(0)
        cells = [f"{d[(M.era == e).values].sum():+.2f}" for e in eras]
        bi = C.index[C.block == "B25"]
        cells.append(f"{(P.loc[bi, v] - b.loc[bi]).sum():+.2f}")
        W(f"| {v} | " + " | ".join(cells) + " |")
        for e in eras:
            BT.append(dict(variant=v, era=e, N=int((M.era == e).sum()), d_sum=d[(M.era == e).values].sum()))
    pd.DataFrame(BT).to_csv(OUT + "_variant_by_batch.csv", index=False)

    # ── §F by state
    W("\n## F. By state — every variant applied ONLY inside one state (Δ = 0 elsewhere), KEEP master\n")
    W(f"States: BTC regime STRONG_BULL vs rest (entry stamp) · entry ATR % terciles (lo < {alo} ≤ mid < {ahi} ≤ hi). "
      f"Same Bonferroni K = {Kt}. Shown: every state read with Δ > 0 and nominal 95 % CI above 0, then the state split of the "
      "top-5 whole-cohort variants.\n")
    srows = []
    for v in READ:
        for st in RN.STATES:
            m_ = {"SBULL": M.sbull, "notSBULL": ~M.sbull}.get(st, M.atr_b == st).values
            d = (P.loc[M.index, v] - b.loc[M.index]).fillna(0.0).where(m_, 0.0)
            lo, hi, p0 = boot(d.values, M.day.values)
            blo, bhi, _ = boot(d.values, M.day.values, alpha=alpha_b)
            dz = d[m_]
            ge = M.era_group.values
            gpos = sum(1 for e in DD.GROUP_ORDER[:-1] if (m_ & (ge == e)).sum() >= 3 and dz[(ge[m_] == e)].mean() > 1e-9)
            gn = sum(1 for e in DD.GROUP_ORDER[:-1] if (m_ & (ge == e)).sum() >= 3)
            tot = d.sum()
            srows.append(dict(variant=v, state=st, n_state=int(m_.sum()), d_in_state=dz.mean(), d_cohort=d.mean(), d_sum=tot, lo95=lo, hi95=hi,
                              p_le0=p0, lo_bonf=blo, hi_bonf=bhi, groups_pos=gpos, groups_n=gn,
                              d_recent=d[np.isin(ge, ["G5 B12-B16", "G6 B17-B24"])].mean(),
                              top_day_share=(d.groupby(M.day.values).sum().max() / tot) if tot > 0 else np.nan,
                              base_in_state=b.loc[M.index][m_].mean(), after_in_state=P.loc[M.index, v][m_].mean()))
    SS = pd.DataFrame(srows)
    SS.to_csv(OUT + "_state_summary.csv", index=False)
    W("| variant @ state | fills in state | in-state BASE → after | Δ/fill cohort [95 % CI] | Bonf. lo | P(Δ≤0) | groups + | recent Δ | top-day share |")
    W("|---|---|---|---|---|---|---|---|---|")
    sel = SS[(SS.lo95 > 0)].sort_values("d_cohort", ascending=False)
    for r in sel.itertuples():
        W(f"| {r.variant} @ {r.state} | {r.n_state} | {r.base_in_state:+.3f} → {r.after_in_state:+.3f} | {r.d_cohort:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}] | "
          f"{r.lo_bonf:+.3f} | {r.p_le0:.3f} | {r.groups_pos}/{r.groups_n} | {r.d_recent:+.3f} | {r.top_day_share * 100:.0f}% |")
    if not len(sel):
        W("| (none) | | | | | | | | |")
    # full gate legs for any state read that clears the Bonferroni lower bound
    Yv0 = pd.read_csv(OUT + "_yr5_variants.csv.gz"); YP0 = Yv0.pivot(index="key", columns="variant", values="pct")
    Y0 = RP.load_yr().set_index("key"); Y0 = Y0.loc[Y0.index.intersection(YP0.index)]
    Y0["day"] = Y0.opened_at.astype(str).str[:10]
    ya = pd.to_numeric(Y0.entry_atr_pct, errors="coerce")
    ystate = {"SBULL": Y0.entry_btc_regime.astype(str).eq("STRONG_BULL"), "notSBULL": ~Y0.entry_btc_regime.astype(str).eq("STRONG_BULL"),
              "ATR_lo": ya < alo, "ATR_mid": (ya >= alo) & (ya < ahi), "ATR_hi": ya >= ahi}
    for r in SS[SS.lo_bonf > 0].itertuples():
        m_ = {"SBULL": M.sbull, "notSBULL": ~M.sbull}.get(r.state, M.atr_b == r.state).values
        d = (P.loc[M.index, r.variant] - b.loc[M.index]).fillna(0.0)
        dz = d[m_]; gz = M[m_]
        W(f"\n**Gate legs — {r.variant} @ {r.state}** (passes the Bonferroni bound): in-state {nwr(b.loc[gz.index])} → "
          f"{nwr(P.loc[gz.index, r.variant])}; Δ per in-state fill {dz.mean():+.3f} (haircut {dz.mean() * 0.5:+.3f}…{dz.mean() * 0.7:+.3f}); "
          f"Σ {dz.sum():+.2f} % pts ≈ ${dz.sum() / 100 * REF_NOTIONAL:+,.0f} at today's ticket over {gz.day.nunique()} days; "
          f"changed {int((dz.abs() > 1e-9).sum())} (up {int((dz > 1e-9).sum())} / down {int((dz < -1e-9).sum())}); "
          f"per group: " + ", ".join(f"{e} {dz[(gz.era_group == e).values].mean():+.3f} ({int((gz.era_group == e).sum())})"
                                    for e in DD.GROUP_ORDER[:-1] if (gz.era_group == e).any())
          + f"; ex-washed {dz[~gz.washed.astype(bool).values].mean():+.3f}; top day {conc(dz.groupby(gz.day.values).sum().max() / dz.sum())}, "
          f"top pair {conc(dz.groupby(gz.pair.values).sum().max() / dz.sum())}; B25 in-state "
          f"{(P.loc[C.index[(C.block == 'B25').values & ({'SBULL': C.sbull, 'notSBULL': ~C.sbull}.get(r.state, C.atr_b == r.state)).values], r.variant] - b.loc[C.index[(C.block == 'B25').values & ({'SBULL': C.sbull, 'notSBULL': ~C.sbull}.get(r.state, C.atr_b == r.state)).values]]).describe()[['count', 'mean']].round(3).to_dict()}.")
        ym = ystate[r.state].values
        yd = (YP0.loc[Y0.index, r.variant] - YP0.loc[Y0.index, "BASE"]).fillna(0)[ym]
        lo_, hi_, _ = boot(yd.values, Y0.day.values[ym], reps=5000)
        W(f"  yr5 backtest in-state: N {ym.sum()} · Δ {yd.mean():+.4f} [{lo_:+.4f}, {hi_:+.4f}] · seeds "
          + "/".join(f"{yd[(Y0.seed.values[ym] == s)].mean():+.4f}" for s in sorted(Y0.seed.unique())) + ".")
    W("\nState split of the top-5 whole-cohort variants (in-state Δ per fill):\n")
    W("| variant | " + " | ".join(RN.STATES) + " |"); W("|---|" + "---|" * len(RN.STATES))
    for v in top[:5]:
        W(f"| {v} | " + " | ".join(f"{SS[(SS.variant == v) & (SS.state == s)].n_state.iloc[0]}: "
                                    f"{SS[(SS.variant == v) & (SS.state == s)].d_in_state.iloc[0]:+.3f}" for s in RN.STATES) + " |")

    # ── §G backtest cross-check
    W("\n## G. Backtest cross-check — the same grid on the yr5 replay fills (today's config, 3 seeds, ex chop∧burst)\n")
    Yv = pd.read_csv(OUT + "_yr5_variants.csv.gz")
    YP = Yv.pivot(index="key", columns="variant", values="pct")
    for p, h in RN.PARTIALS.items():
        YP[p] = 0.5 * YP["BASE"] + 0.5 * YP[h]
    Y = RP.load_yr().set_index("key")
    Y = Y.loc[Y.index.intersection(YP.index)]
    Y["day"] = Y.opened_at.astype(str).str[:10]; Y["H"] = np.where(Y.opened_at.astype(str).str[:7] <= "2026-04", "H1", "H2")
    W(f"yr5 BASE through this evaluator: {nwr(YP.loc[Y.index, 'BASE'])} (the 10-05 study: 1679·62 %·−0.057 — must match).\n")
    yrows = []
    for v in READ:
        d = (YP.loc[Y.index, v] - YP.loc[Y.index, "BASE"]).fillna(0)
        lo, hi, p0 = boot(d.values, Y.day.values, reps=5000)
        yrows.append(dict(variant=v, N=len(d), d=d.mean(), lo95=lo, hi95=hi, dH1=d[(Y.H == "H1").values].mean(),
                          dH2=d[(Y.H == "H2").values].mean(),
                          seeds="/".join(f"{d[(Y.seed == s).values].mean():+.3f}" for s in sorted(Y.seed.unique()))))
    YS = pd.DataFrame(yrows)
    YS = YS.merge(SM[["variant", "d", "verdict"]].rename(columns={"d": "d_master"}), on="variant")
    YS["agree"] = np.sign(YS.d.round(4)) == np.sign(YS.d_master.round(4))
    YS.to_csv(OUT + "_yr5_summary.csv", index=False)
    W("| variant | master Δ | yr5 Δ [95 % CI] | yr5 H1 / H2 | seeds | same sign | yr5 significant opposite? |"); W("|---|---|---|---|---|---|---|")
    for r in YS.sort_values("d_master", ascending=False).itertuples():
        opp = (r.d_master > 0 and r.hi95 < 0) or (r.d_master < 0 and r.lo95 > 0)
        W(f"| {r.variant} | {r.d_master:+.3f} | {r.d:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}] | {r.dH1:+.3f} / {r.dH2:+.3f} | {r.seeds} | "
          f"{'yes' if r.agree else 'NO'} | {'YES' if opp else 'no'} |")

    open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
    # per-fill table
    keep = ["block", "era", "era_group", "pair", "opened_at", "closed_at", "reason_live", "y_trad", "y", "peak_pnl", "trough_pnl", "src",
            "mfe", "mfe_min", "mae", "mae_min", "mae_before_arm", "peak_by5", "peak_by10", "peak_by15", "post5", "post15", "post30", "post60",
            "postmax60", "runpb0.3", "runpb0.5", "runpb1.0", "runpb0.5_cens", "entry_btc_regime", "atr", "atr_b", "washed"]
    out = C[[c for c in keep if c in C.columns]].copy()
    out["BASE_replica"] = P["BASE"]; out["BASE_reason"] = R["BASE"]
    out.to_csv(OUT + "_postexit.csv")
    print(f"tables → {OUT}_tables.md · K = {Kt}")
    print(SM.sort_values("d", ascending=False)[["variant", "d", "lo95", "hi95", "lo_bonf", "groups_pos", "groups_n", "d_G5", "d_G6",
                                                "d_ex_washed", "top_day_share", "verdict"]].to_string())


if __name__ == "__main__":
    main()
