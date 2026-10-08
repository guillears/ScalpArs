#!/usr/bin/env python3
"""MOMENTUM-LONG recall trace (2026-10-08) — step 3: per-signal final class, pricing on matched fills, the gap waterfall and the
calibrated per-trade estimate. Read-only.

Inputs: reports/study_ml_trace_matchset.csv (build), reports/study_ml_trace_seedrows.csv + study_ml_trace_extras.csv (eval), the yr5
fills (scripts/yr5_fills_trimmed.py) and reports/MASTER_POOL_stacked.csv (forward live read).
Output: reports/study_ml_trace_signals.csv (one row per live ML fill: final class + per-seed classes) and every table of
reports/MOMENTUM_LONG_RECALL_TRACE_2026-10-08.md on stdout.
Usage: venv/bin/python scripts/study_ml_trace_report.py
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yr5_fills_trimmed as YT                                          # noqa: E402

REP = os.path.join(ROOT, "reports")
USD = 4.875 * 3000 / 100.0          # $ per 1 %-point on a fixed $3k book at 1× (yr5_fills_trimmed.fixed_book_usd, cells off)
rng = np.random.default_rng(7)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)


def fm(x):
    x = pd.Series(x).dropna()
    return f"{len(x)} · {(x > 0).mean() * 100:.0f}% · {x.mean():+.3f}" if len(x) else "0"


def boot(df, col="pct", day="t", n=4000, per_seed=False):
    """day-block bootstrap 95 % CI of the mean."""
    d = df.assign(_d=pd.to_datetime(df[day]).dt.floor("D"))
    g = d.groupby("_d")[col].agg(["sum", "size"])
    s, c = g["sum"].values, g["size"].values
    k = len(g)
    idx = rng.integers(0, k, (n, k))
    m = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


def era_group(e):
    return {"BASE": "1 BASE Jun15-Jul10", "B1": "2 B1 Jul11-31", "B2": "3 B2-B3 Aug1-24", "B3": "3 B2-B3 Aug1-24",
            "B4": "4 B4-B5 Aug24-27", "B5": "4 B4-B5 Aug24-27"}.get(
        e, "5 B6-B12 Sep11-25" if e in ("B6", "B7", "B8", "B9", "B10", "B12") else "6 B13-B16 Sep26-Oct3")


def main():
    M = pd.read_csv(os.path.join(REP, "study_ml_trace_matchset.csv"), low_memory=False); M["t"] = pd.to_datetime(M.t)
    S = pd.read_csv(os.path.join(REP, "study_ml_trace_seedrows.csv")); S["t"] = pd.to_datetime(S.t)
    E = pd.read_csv(os.path.join(REP, "study_ml_trace_extras.csv")); E["t"] = pd.to_datetime(E.t)
    L = M[M.side == "LIVE"].copy(); R = M[M.side == "REPLAY"].copy()
    for c in ("matched_live", "live_up"):
        R[c] = R[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    L["kept"] = L.stack_keep.astype(str) == "True"
    L["grp"] = np.where(L.kept, "kept", np.where(L.in_master.astype(str) == "True", "removed", "not_in_master"))
    L["ns"] = L.n_seeds.fillna(0).astype(int)
    L["G"] = L.era.map(era_group)
    F = YT.load(sleeves=["MOM-long"])
    # ── V1 validation against the master / the replay outputs (validate_against_master spirit) ──
    Pm0 = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
    mk = Pm0[(Pm0.entry_strategy.fillna("MOMENTUM") == "MOMENTUM") & (Pm0.direction == "LONG") & (Pm0.status == "CLOSED")
             & ~Pm0.cell_multiplier_source.fillna("").str.contains("PROBE") & (Pm0.stack_keep == True)
             & (Pm0.opened_at.astype(str) < "2026-10-04") & ~Pm0.era.isin(["BASE", "B1"])]
    lk = set(zip(L.t.dt.strftime("%Y-%m-%dT%H:%M:%S"), L.pair))
    miss = [(a, b) for a, b in zip(mk.opened_at.astype(str).str[:19].str.replace(" ", "T"), mk.pair) if (a, b) not in lk]
    eq = {(int(s), p, t.strftime("%Y-%m-%d %H:%M:%S")): v for s, p, t, v in zip(F.seed, F.pair, F.t, F.pct)}
    bad = sum(abs(eq.get((int(s), p, t.strftime("%Y-%m-%d %H:%M:%S")), 9e9) - v) > 1e-9 for s, p, t, v in zip(R.seed, R.pair, R.t, R.pct_rep))
    lv = {(a, b): v for a, b, v in zip(Pm0.opened_at.astype(str).str[:19].str.replace(" ", "T"), Pm0.pair, pd.to_numeric(Pm0.pnl_percentage, errors="coerce"))}
    chk = [(abs(lv[(a, b)] - v) < 1e-9) for a, b, v in zip(L.t.dt.strftime("%Y-%m-%dT%H:%M:%S"), L.pair, L.pct_live_raw) if (a, b) in lv]
    print(f"V1 master-kept full-size ML B2→Oct-3 missing from the live set: {len(miss)} {miss[:3]} · replay rows ≠ yr5 fills: {bad}/{len(R)}"
          f" · live rows in master with equal pct: {sum(chk)}/{len(chk)} → {'PASS' if not miss and not bad and all(chk) else 'FAIL'}")
    eras = L.groupby("era").t.min().sort_values()

    def era_of(t):
        k = int(np.searchsorted(eras.values.astype("datetime64[ns]"), np.datetime64(t), side="right")) - 1
        return eras.index[max(0, k)]
    R["era"] = [era_of(t) for t in R.t]; R["G"] = R.era.map(era_group)
    RU = R[R.live_up]

    # ── per-signal final class ──
    S["fam"] = S.cls.str.replace("GATE:", "", regex=False)
    rows = []
    for i, x in L.iterrows():
        s = S[S.live_idx == i]
        miss = s[s.cls != "MATCHED"]
        modal = miss.fam.value_counts().index[0] if len(miss) else ""
        gate1 = miss.detail.astype(str).str.split("+").str[0].value_counts().index[0] if len(miss) else ""
        lc = (x.asw_kind == "MATCHED") or (x.lp_kind == "MATCHED")
        if x.ns == 3:
            fc = "R3 reproduced in all 3 seeds"
        elif x.ns > 0:
            fc = "R12 reproduced in 1-2 seeds (other seeds: scan phase)"
        elif x.grp == "removed":
            fc = "E today's rules refuse it (master-removed)"
        elif x.grp == "not_in_master":
            fc = "E0 never in master (older pre-screen)"
        elif lc:
            fc = "P live-clock replays take it, yr5 phase never does"
        else:
            fc = {"BTC_MARKET": "G1 BTC forming-bar gate", "PAIR_CANDLE": "G2 pair forming-candle gate",
                  "TODAY_RULE": "G3 today-rule gate at yr5's second (master keeps)", "STATE_ROUTING": "G4 door/routing/state gate",
                  "DIR_SHORT": "G5 replay read the pair SHORT", "NO_CANDIDATE": "G6 no LONG candidate at yr5's scans",
                  "CAPACITY": "C capacity (yr5 book)", "TOOK_OTHER_TIME": "T replay took it at another time",
                  "PAIR_HELD": "C capacity (yr5 book)", "NOT_SCANNED": "U not scanned"}.get(modal, "X other")
        rows.append(dict(live_idx=i, era=x.era, pair=x.pair, opened_at=x.t, grp=x.grp, stack_reason=x.stack_reason,
                         pct_live=x.pct_live_raw, pct_live_stack=x.pct_live_stack, live_exit=x.close_reason, cell=x.cell,
                         n_seeds=x.ns, pct_rep_mean=x.pct_rep, rep_exit=x.rep_reason, asw=x.asw_kind, asw_cls=x.asw_cls, lp=x.lp_kind,
                         final_class=fc, modal_seed_family=modal, modal_gate=gate1,
                         seed_classes=" | ".join(f"s{int(r.seed)}:{r.cls}{'(' + str(r.detail)[:40] + ')' if r.cls != 'MATCHED' else ''}"
                                                 for r in s.sort_values("seed").itertuples()),
                         med_d_btc_rsi=miss.d_btc_rsi.median() if len(miss) else np.nan,
                         med_d_rsi=miss.d_rsi.median() if len(miss) else np.nan,
                         tick_src=",".join(sorted(set(s.src.astype(str))))))
    SG = pd.DataFrame(rows)
    SG.to_csv(os.path.join(REP, "study_ml_trace_signals.csv"), index=False)

    print("## §1 matched set")
    print(f"live full-size ML (Jun-18 → Oct-3) {fm(L.pct_live_raw)} · kept {fm(L[L.kept].pct_live_stack)} · removed {fm(L[L.grp == 'removed'].pct_live_raw)}"
          f" · not in master {fm(L[L.grp == 'not_in_master'].pct_live_raw)}")
    for g in ("kept", "removed", "not_in_master"):
        d = L[L.grp == g]
        print(f"  {g:14s} recall ≥1 seed {(d.ns > 0).mean() * 100:3.0f}% · per seed {d.ns.sum() / 3 / len(d) * 100:3.0f}% · all 3 {(d.ns == 3).mean() * 100:3.0f}%"
              f" · asw {(d.asw_kind == 'MATCHED').mean() * 100:3.0f}% · lp {(d.lp_kind == 'MATCHED').mean() * 100:3.0f}%")
    print(f"replay ML Jun-15→Oct-4: {len(R[R.t >= '2026-06-15']) / 3:.1f}/seed · live-up {len(RU) / 3:.1f}/seed {fm(RU.pct_rep)} CI {boot(RU.rename(columns={'pct_rep': 'pct'}))}"
          f" · live-down {fm(R[(~R.live_up) & (R.t >= '2026-06-15')].pct_rep)}")
    print(f"  kept CI {boot(L[L.kept].rename(columns={'pct_live_stack': 'pct'}))} · as traded CI {boot(L.rename(columns={'pct_live_raw': 'pct'}))}")
    print("\n## by n_seeds (live)"); print(L.groupby(["grp", "ns"]).pct_live_raw.agg(["size", "mean", lambda x: (x > 0).mean()]).round(3))

    print("\n## §2 per-signal final class (live fills)")
    t = SG.groupby("final_class").agg(n=("pct_live", "size"), wr=("pct_live", lambda x: (x > 0).mean() * 100),
                                      live=("pct_live", "mean"), live_stack=("pct_live_stack", "mean"), rep=("pct_rep_mean", "mean"))
    print(t.round(3).to_string())
    k0 = SG[(SG.grp == "kept") & (SG.n_seeds == 0)]
    print("\nkept never reproduced — modal gate:"); print(k0.groupby(["final_class", "modal_gate"]).pct_live.agg(["size", "mean"]).round(3).to_string())
    rem = SG[SG.grp != "kept"]
    print("\nremoved / not-in-master by stack reason × yr5 modal family:"); print(pd.crosstab(rem.stack_reason.fillna("(not in master)"), rem.modal_seed_family))
    print("\nseed-level live-only (kept) classes:")
    sk = S[(S.cls != "MATCHED") & (S.stack_keep.astype(str) == "True")]
    print(sk.groupby("cls").agg(n=("pct_live", "size"), live=("pct_live", "mean")).round(3).to_string())
    sk1 = sk.assign(g=sk.detail.astype(str).str.split("+").str[0])
    print(sk1.groupby(["cls", "g"]).agg(n=("pct_live", "size"), live=("pct_live", "mean")).sort_values("n", ascending=False).head(25).round(3).to_string())
    print("\nreplay − live readings at the replay's nearest scan (live-only seed rows): BTC RSI median |Δ| "
          f"{S.d_btc_rsi.abs().median():.2f}, mean {S.d_btc_rsi.mean():+.2f}; pair RSI median |Δ| {S.d_rsi.abs().median():.2f} mean {S.d_rsi.mean():+.2f};"
          f" pair ADX median |Δ| {S.d_adx.abs().median():.2f}; price median {S.d_price_pct.median():+.3f}%")
    print("tick source of every traced (signal × seed):", S.src.value_counts().to_dict())
    F["tk"] = [os.path.exists(os.path.join(REP, "backtest_cache", "ticks_q", p, f"{t:%Y-%m-%d}.npz")) for p, t in zip(F.pair, F.t)]
    print(f"yr5 ML fills (year) on a pair-day with a tick archive: {F.tk.mean() * 100:.2f}% of {len(F)}")

    print("\n## §3 replay-only (live up) classes")
    E["g1"] = E.asw_gate.fillna("").astype(str).str.split("+").str[0].str.split("[").str[0].str.replace("MACRO:", "", regex=False)
    t = E.groupby("cls").agg(per_seed=("pct_rep", lambda x: len(x) / 3), wr=("pct_rep", lambda x: (x > 0).mean() * 100),
                             avg=("pct_rep", "mean"), sum_per_seed=("pct_rep", lambda x: x.sum() / 3))
    t["usd_per_seed_1x"] = t.sum_per_seed * USD
    print(t.round(3).to_string())
    print(E.groupby(["cls", "cell"]).pct_rep.agg(["size", "mean"]).round(3).to_string())
    print(E.assign(m=E.t.dt.to_period("M")).pivot_table(index="m", columns="cls", values="pct_rep", aggfunc="size").fillna(0).astype(int))
    print(E[E.cls.isin(["RULE_THEN", "PHASE"])].groupby(["cls", "g1"]).pct_rep.agg(["size", "mean"]).sort_values("size", ascending=False).head(24).round(3).to_string())

    print("\n## §4 same-signal pricing (matched, per seed)")
    rows = []
    for i, x in L.iterrows():
        for s in (1, 2, 3):
            c = F[(F.seed == s) & (F.pair == x.pair) & ((F.t - x.t).abs() <= pd.Timedelta(minutes=10))]
            if len(c):
                y = c.iloc[(c.t - x.t).abs().argmin()]
                rows.append(dict(grp=x.grp, G=x.G, live=x.pct_live_raw, rep=y.pct, dt=(y.t - x.t).total_seconds(),
                                 lex=str(x.close_reason).split(" ")[0], rex=str(y.close_reason).split(" L")[0].split(" ")[0],
                                 dpx=(float(y.entry_price) / float(x.entry_price) - 1) * 100,
                                 lot=x.entry_order_type, rot=y.entry_order_type))
    P = pd.DataFrame(rows); P["d"] = P.rep - P.live
    k = P[P.grp == "kept"]
    print(f"kept matched seed-pairs {len(k)}: live {k.live.mean():+.3f} vs replay {k.rep.mean():+.3f} (Δ {k.d.mean():+.3f}) · same exit {(k.lex == k.rex).mean() * 100:.0f}%"
          f" · Δ when same exit {k[k.lex == k.rex].d.mean():+.3f} (n {int((k.lex == k.rex).sum())}) · open lag median {k.dt.median():+.0f}s · |lag|≤30 s {(k.dt.abs() <= 30).mean() * 100:.0f}%"
          f" · entry px median {k.dpx.median():+.3f}% · MAKER live {(k.lot == 'MAKER').mean() * 100:.0f}% / replay {(k.rot == 'MAKER').mean() * 100:.0f}%")
    k = k.assign(why=np.select(
        [k.lex == k.rex,
         k.rex.str.startswith("RH_") | (k.rex == "HARD_TP_LADDER") & (k.lex != "HARD_TP_LADDER") | (k.lex == "TRAILING_STOP") | (k.lex == "MANUAL"),
         (k.dt.abs() > 30)],
        ["same exit", "exit rule differs (today's ladder / recovery hold vs the era's exits)", "entry at a different scan (> 30 s apart)"],
        "same scan, knife-edge (stop width / runner arm)"))
    print(k.groupby("why").agg(n=("d", "size"), live=("live", "mean"), rep=("rep", "mean"), d=("d", "mean"), d_sum=("d", "sum")).round(3).to_string())
    for e in ("STOP_LOSS", "STOP_LOSS_WIDE"):
        b = k[(k.lex == e) & (k.rex == e)]
        print(f"  both {e}: {len(b)} live {b.live.mean():+.3f} replay {b.rep.mean():+.3f}  (stop-fill overshoot)")
    print(P.groupby(["G"]).agg(n=("d", "size"), live=("live", "mean"), rep=("rep", "mean"), d=("d", "mean")).round(3).to_string())

    print("\n## §5 per-era: live as traded / live kept / yr5 (live-up hours)")
    out = []
    for G in sorted(L.G.unique()):
        l = L[L.G == G]; kk = l[l.kept]; r = RU[RU.G == G]
        out.append(dict(era=G, live=fm(l.pct_live_raw), kept=fm(kk.pct_live_stack), rep_per_seed=f"{len(r) / 3:.1f} · {(r.pct_rep > 0).mean() * 100:.0f}% · {r.pct_rep.mean():+.3f}"))
    print(pd.DataFrame(out).to_string())
    # waterfall on live-up hours (per seed averages)
    print("\n## §5 waterfall: live kept (+) → yr5 live-up")
    W = []
    for s in (1, 2, 3):
        rs = RU[RU.seed == s]
        mrep = rs[rs.matched_live]
        only = E[E.seed == s]
        kl = S[(S.seed == s) & (S.stack_keep.astype(str) == "True")]
        km = kl[kl.cls == "MATCHED"]; ko = kl[kl.cls != "MATCHED"]
        W.append(dict(seed=s, kept_n=len(kl), kept_sum=kl.pct_live_stack.sum(),
                      kept_matched_n=len(km), kept_matched_live=km.pct_live_stack.sum(),
                      kept_only_n=len(ko), kept_only_sum=ko.pct_live_stack.sum(),
                      rep_n=len(rs), rep_sum=rs.pct_rep.sum(), rep_matched_n=len(mrep), rep_matched_sum=mrep.pct_rep.sum(),
                      **{f"only_{c}_n": int((only.cls == c).sum()) for c in sorted(E.cls.unique())},
                      **{f"only_{c}_sum": only[only.cls == c].pct_rep.sum() for c in sorted(E.cls.unique())}))
    W = pd.DataFrame(W)
    print(W.mean(numeric_only=True).round(3).to_string())

    print("\n## §6 forward live read (master, full-size ML)")
    Pm = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
    m = Pm[(Pm.entry_strategy.fillna("MOMENTUM") == "MOMENTUM") & (Pm.direction == "LONG") & (Pm.status == "CLOSED")
           & ~Pm.cell_multiplier_source.fillna("").str.contains("PROBE")].copy()
    m["pct"] = pd.to_numeric(m.pnl_percentage); m["sp"] = pd.to_numeric(m.stack_pct); m["t"] = pd.to_datetime(m.opened_at.str[:19])
    for lo, lab in (("2026-06-01", "all master eras"), ("2026-07-03", "ex washed-out window (from Jul-3)"), ("2026-08-01", "Aug-1 →"),
                    ("2026-09-11", "Sep-11 →"), ("2026-09-29", "Sep-29 → (LOADX live)")):
        d = m[m.t >= lo]; kk = d[d.stack_keep == True]
        print(f"  {lab:34s} as traded {fm(d.pct)} · kept {fm(kk.sp)}")
    rr = R[(R.t >= "2026-09-11")]
    print(f"  yr5 Sep-11 → Oct-4 all hours {len(rr) / 3:.1f}/seed {fm(rr.pct_rep)} · live-up {fm(rr[rr.live_up].pct_rep)}")
    for lab, d in (("yr5 YEAR", F), ("yr5 H1 Jan-Apr", F[F.t < "2026-05-01"]), ("yr5 May-Oct", F[F.t >= "2026-05-01"]),
                   ("yr5 Jun-15→Oct-4 live-up", F[F.index.isin([])]),):
        if len(d):
            print(f"  {lab:28s} {len(d) / 3:.1f}/seed {fm(d.pct)} CI {boot(d)}")
    print(f"  yr5 live-up Jun-15→Oct-4 ex BASE: {fm(RU[RU.era != 'BASE'].pct_rep)} · BASE {fm(RU[RU.era == 'BASE'].pct_rep)}")
    print(f"  live kept ex BASE {fm(L[L.kept & (L.era != 'BASE')].pct_live_stack)} · ex BASE ex B1 {fm(L[L.kept & ~L.era.isin(['BASE', 'B1'])].pct_live_stack)}")


if __name__ == "__main__":
    main()
