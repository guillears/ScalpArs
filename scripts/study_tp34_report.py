#!/usr/bin/env python3
"""🎯 Oct-8 study (+3 vs +4 tick check) — tables from the tick walks (study_tp34_walk.py, study_tp34_willy.py) → scratch tp34/tables.md
+ reports/FRENZY_TP3_VS_TP4_TICKS_2026-10-08_fills.csv (one row per cohort fill, every exit) + ..._willy.csv.
Usage: venv/bin/python scripts/study_tp34_report.py"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_tp34_common as C                                   # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = os.path.join(SCR, "tp34"); REP = os.path.join(ROOT, "reports")
TS = pd.Timestamp("2026-05-20"); T0, T1 = pd.Timestamp("2026-01-04"), pd.Timestamp("2026-10-04")
BOOK, SLOT = 3000.0, 0.975 / 4
EX = ["fix3", "fix4", "fix5", "fix6", "lock", "trail5"]
L = []


def P(s=""):
    L.append(s)


def load():
    A = pd.read_pickle(os.path.join(OUT, "cohort_a.pkl")); B = pd.read_pickle(os.path.join(OUT, "cohort_b.pkl"))
    Cc = pd.read_pickle(os.path.join(OUT, "cohort_c.pkl"))
    out = {}
    for k, D in (("a", A), ("b", B), ("c", Cc)):
        W = pd.read_pickle(os.path.join(OUT, f"walk_{k}.pkl"))
        D = D.assign(key=D.pair + "|" + D.sig.astype("int64").astype(str)).merge(W, on="key", how="left")
        D["t"] = pd.to_datetime(D.sig, unit="ms"); D["day"] = D.t.dt.floor("D"); D["month"] = D.t.dt.strftime("%Y-%m")
        D["half"] = np.where(D.t < TS, "H1", "H2")
        out[k] = D
    return out


def ci(v, d):
    lo, hi = C.boot_day(v, d)
    return f"[{lo:+.3f}, {hi:+.3f}]"


def exit_table(D, title, pre=""):
    P(f"**{title}** — N {len(D)} fills · {D.day.nunique()} days")
    P()
    P("| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |")
    P("|---|---|---|---|---|---|---|---|---|")
    for x in EX:
        v = D[pre + x]
        h1, h2 = D.half == "H1", D.half == "H2"
        if x == "fix3":
            dl = "–"; mo = "–"; wt = "–"
        else:
            d = v - D[pre + "fix3"]
            dl = f"{d.mean():+.3f} {ci(d.values, D.day.values)}"
            mm = d.groupby(D.month).mean(); mo = f"{(mm > 0).sum()} of {len(mm)}"
            wt = f"{d.drop(d.nlargest(5).index).mean():+.3f}"
        P(f"| {C.LABEL[x]} | {(v > 0).mean() * 100:.0f} % | **{v.mean():+.3f}** | {v.sum():+.0f} | {v[h1].mean():+.3f} ({h1.sum()}) | "
          f"{v[h2].mean():+.3f} ({h2.sum()}) | {dl} | {mo} | {wt} |")
    P()


def month_signs(D, pre=""):
    d4 = (D[pre + "fix4"] - D[pre + "fix3"]).groupby(D.month).agg(["mean", "count"])
    dl = (D[pre + "lock"] - D[pre + "fix3"]).groupby(D.month).mean()
    P("| month | N | Δ(+4 − +3) | Δ(lock − +3) |"); P("|---|---|---|---|")
    for m, r in d4.iterrows():
        P(f"| {m} | {int(r['count'])} | {r['mean']:+.3f} | {dl[m]:+.3f} |")
    P()


def main():
    Z = load()
    A, B, Cc = Z["a"], Z["b"], Z["c"]
    A = A[A.st == "ok"]; ns = A.seed.nunique()
    # ═════ (a) ═════
    P("## (a) yr5 replay FRENZY_LONG + FRENZY_WIDE, today's entry stack (3 seeds pooled, day clusters)"); P()
    exit_table(A, "(a) all kept fills, live ruler (8 s + 0.10 % entry slip)")
    for s in ("FRENZY_LONG", "FRENZY_WIDE"):
        exit_table(A[A.sleeve == s], f"(a) {s}")
    P("Per seed (replicates; seeds differ only in replay tick ordering → slot occupancy):"); P()
    P("| seed | N | +3/−3 | +4/−3 | Δ +4−+3 | lock | Δ lock−+3 | +6/−3 |"); P("|---|---|---|---|---|---|---|---|")
    for sd, x in A.groupby("seed"):
        P(f"| {sd} | {len(x)} | {x.fix3.mean():+.3f} | {x.fix4.mean():+.3f} | {(x.fix4 - x.fix3).mean():+.3f} | {x.lock.mean():+.3f} | {(x.lock - x.fix3).mean():+.3f} | {x.fix6.mean():+.3f} |")
    P()
    exit_table(A, "(a) sensitivity: same fills, 8 s, NO entry slip", "ns_")
    P("Per-month Δ (a, live ruler):"); P(); month_signs(A)
    # why the peak approximation disagreed
    P("### Why the peak-based approximation disagreed (cohort a, same fills)"); P()
    ap = np.where(A.rep_pk >= 3, 3.0, A.rep_pct)
    rows = [("replay as run (its exit = fixed +4/−3)", A.rep_pct.mean(), "–"),
            ("peak approximation of +3 (replay peak ≥ 3 → +3, else replay result) — the halves table", ap.mean(), "–"),
            ("ticks +4/−3 (live ruler)", A.fix4.mean(), "–"), ("ticks +3/−3 (live ruler)", A.fix3.mean(), "–")]
    P("| pricing | avg %/fill |"); P("|---|---|")
    for n, v, _ in rows:
        P(f"| {n} | {v:+.3f} |")
    P()
    r4 = A.rep_reason == "FRENZY_TP"; t4 = A.fix4_how == "TP"; t3 = A.fix3_how == "TP"
    P(f"- replay +4 TP rate {r4.mean() * 100:.1f} % · ticks +4 TP rate {t4.mean() * 100:.1f} % · ticks +3 TP rate {t3.mean() * 100:.1f} %")
    P(f"- replay peak ≥ +3 {(A.rep_pk >= 3).mean() * 100:.1f} % · ticks peak-before-stop ≥ +3 {(A.pk_before_stop >= 3).mean() * 100:.1f} % · ≥ +4 {(A.pk_before_stop >= 4).mean() * 100:.1f} %")
    P(f"- replay vs ticks +4/−3 outcome agreement (TP/SL/CAP): {((A.rep_reason.map({'FRENZY_TP': 'TP', 'STOP_LOSS': 'SL', 'MAX_HOLD_TIME': 'CAP'})) == A.fix4_how).mean() * 100:.1f} %")
    band = (A.pk_before_stop >= 3) & (A.pk_before_stop < 4)
    P(f"- the +3-only 'rescue' band (peak before the stop in [+3, +4)): ticks {band.mean() * 100:.1f} % · replay {((A.rep_pk >= 3) & (A.rep_pk < 4) & (A.rep_reason == 'STOP_LOSS')).mean() * 100:.1f} %")
    P(f"- entry: replay E vs tick-ruler E median {np.nanmedian((A.E / A.rep_E - 1) * 100):+.3f} % (replay opens {((A.rep_open - A.sig) / 1000).median():.1f} s after the close, ruler 8 s + 0.10 % slip)")
    P()
    # ═════ (b) ═════
    P("## (b) the Oct-4/5 850-fill tick cohort"); P()
    P(f"Parity (old ruler, entry = 1-min close after the signal, flat 0.09 fees): +3/−3 {B.old_fix3.mean():+.3f} · +4/−3 {B.old_fix4.mean():+.3f} · "
      f"+6/−3 {B.old_fix6.mean():+.3f} · lock {B.old_lock.mean():+.3f} · trail +5/1.5 {B.old_trail5.mean():+.3f} "
      f"(published: +0.192 · +0.187 · +0.263 · +0.234 · +0.142)"); P()
    d = B.old_fix4 - B.old_fix3
    P(f"Old ruler Δ(+4 − +3) {d.mean():+.3f} {ci(d.values, B.day.values)} · Δ(lock − +3) {(B.old_lock - B.old_fix3).mean():+.3f} {ci((B.old_lock - B.old_fix3).values, B.day.values)}"); P()
    exit_table(B, "(b) all 850, live ruler")
    K = B[B.today_keep]
    exit_table(K, f"(b) today's entry stack (LONG red ATR ≤ 3 + WIDE hold-green, bearish block): {int(K.today_long.sum())} LONG-type · {int(K.today_wide.sum())} WIDE-type")
    P("Per-month Δ (b today's stack, live ruler):"); P(); month_signs(K)
    # ═════ (c) ═════
    P("## (c) FRENZY_LITE 724-fill study cohort"); P()
    Ck = Cc[Cc.st == "ok"]
    exit_table(Ck, "(c) all 724, live ruler")
    exit_table(Ck[~Ck.bear], "(c) bearish-day block applied (= live LITE)")
    P("Per-month Δ (c kept):"); P(); month_signs(Ck[~Ck.bear])
    # ═════ (d) ═════
    D = pd.read_pickle(os.path.join(OUT, "walk_d.pkl")).merge(pd.read_pickle(os.path.join(OUT, "cohort_d.pkl"))[
        ["pair", "live_open", "live_pct", "live_reason", "sleeve"]], on=["pair", "live_open"])
    D["af_era"] = [r[f"af_{r.era}"] for _, r in D.iterrows()]; D["af_era_how"] = [r[f"af_{r.era}_how"] for _, r in D.iterrows()]
    mapr = {"STOP_LOSS": "SL", "FRENZY_TP": "TP", "RUNNER_TRAIL": "TRAIL", "MAX_HOLD_TIME": "CAP"}
    same = (D.live_reason.map(mapr) == D.af_era_how); err = (D.af_era - D.live_pct).abs()
    P("## (d) live master FRENZY-family fills (walker validation)"); P()
    P(f"As filled (live opened_at + entry_price, the exit era in force): **{same.sum()} / {len(D)} same exit reason**, |Δ P&L| max {err.max():.4f} pts, mean {err.mean():.4f}."); P()
    P("| opened (UTC) | pair | sleeve | era exit | live | walker (as filled) | ruler +3/−3 | ruler +4/−3 | ruler lock |"); P("|---|---|---|---|---|---|---|---|---|")
    for _, r in D.sort_values("live_open").iterrows():
        f = lambda v: "open" if pd.isna(v) else f"{v:+.2f}"
        P(f"| {pd.Timestamp(r.live_open, unit='ms'):%m-%d %H:%M} | {r.pair.replace('USDT', '')} | {r.sleeve.replace('FRENZY_', '')} | {C.LABEL[r.era]} | "
          f"{r.live_pct:+.2f} {r.live_reason} | {r.af_era:+.2f} {r.af_era_how} | {f(r.get('fix3'))} | {f(r.get('fix4'))} | {f(r.get('lock'))} |")
    P()
    dd = D.dropna(subset=["fix3", "fix4", "lock"])
    P(f"Live ruler on the {len(dd)} complete live fills: +3/−3 Σ {dd.fix3.sum():+.2f} · +4/−3 Σ {dd.fix4.sum():+.2f} · lock Σ {dd.lock.sum():+.2f} · as traded Σ {dd.live_pct.sum():+.2f}"); P()
    # ═════ halves rows ═════
    P("## Halves-table rows at the live exit (+3/−3/12 h) on ticks"); P()
    P("| Strategy | H1 N · WR · avg % [CI] · $ | H2 N · WR · avg % [CI] · $ | Year N · WR · avg % [CI] · $ |"); P("|---|---|---|---|")
    res = {}
    lite_mult = BOOK * SLOT * max(1, int(round(20 * 0.32))) / 100
    for name, X, w, usd in (("FRENZY_LONG", A[A.sleeve == "FRENZY_LONG"], ns, None), ("FRENZY_WIDE", A[A.sleeve == "FRENZY_WIDE"], ns, None),
                            ("FRENZY_LITE", Ck[~Ck.bear], 1, lite_mult)):
        cells = []; res[name] = {}
        for h, m in (("H1", X.half == "H1"), ("H2", X.half == "H2"), ("FY", X.half.notna())):
            x = X[m & (X.t >= T0) & (X.t < T1)]
            dol = (x.fix3 * (x.usd_per_pct if usd is None else usd)).sum() / w
            lo, hi = C.boot_day(x.fix3.values, x.day.values)
            res[name][h] = dict(N=len(x) / w, WR=(x.fix3 > 0).mean() * 100, avg=x.fix3.mean(), lo=lo, hi=hi, usd=dol)
            cells.append(f"{len(x) / w:.0f} · {(x.fix3 > 0).mean() * 100:.1f} % · **{x.fix3.mean():+.3f}** [{lo:+.2f}, {hi:+.2f}] · {dol:+,.0f} $")
        P(f"| {name} | " + " | ".join(cells) + " |")
    P()
    P("Same rows under the old peak approximation (from YR5_HALVES_TODAY_STACK): FRENZY_LONG FY +0.076 · +$1,870 · FRENZY_WIDE FY +0.288 · +$895 · FRENZY_LITE FY +0.278 · +$6,678.")
    P()
    # ═════ WILLY ═════
    Wl = pd.read_pickle(os.path.join(OUT, "willy_ticks.pkl")); Wl = Wl[Wl.st == "ok"].copy()
    Wl["t"] = pd.to_datetime(Wl.te, unit="ms"); Wl["dayd"] = Wl.t.dt.floor("D"); Wl["half"] = np.where(Wl.t < TS, "H1", "H2")
    wm = BOOK * SLOT * 20 / 100
    P("## FRENZY_WILLY on ticks (bot accounting)"); P()
    P("| variant · trigger | H1 N · WR · avg % [CI] · $ | H2 N · WR · avg % [CI] · $ | Year N · WR · avg % [CI] · $ | worst trade | dipped ≤ −4.5 % | TP / cap / stop |")
    P("|---|---|---|---|---|---|---|")
    for v in ("CONV", "NOSTOP", "BACKSTOP"):
        for tg in ("A", "B", "all"):
            X = Wl[(Wl.variant == v) & ((Wl.trigger == tg) if tg != "all" else True)]
            X = X[(X.t >= T0) & (X.t < T1)]
            cells = []
            for h, m in (("H1", X.half == "H1"), ("H2", X.half == "H2"), ("FY", X.half.notna())):
                x = X[m]; lo, hi = C.boot_day(x.pct.values, x.dayd.values)
                cells.append(f"{len(x)} · {(x.pct > 0).mean() * 100:.1f} % · **{x.pct.mean():+.3f}** [{lo:+.2f}, {hi:+.2f}] · {x.pct.sum() * wm:+,.0f} $")
                if tg == "all":
                    res.setdefault(f"WILLY_{v}", {})[h] = dict(N=len(x), avg=x.pct.mean(), usd=x.pct.sum() * wm)
            vc = X.why.value_counts()
            P(f"| {v} · {tg} | " + " | ".join(cells) + f" | {X.pct.min():+.2f} | {(X.worst <= -4.5).mean() * 100:.1f} % | {vc.get('tp', 0)} / {vc.get('time_cap', 0)} / {vc.get('stop', 0)} |")
    P()
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(OUT, "halves_rows.json"), "w"), indent=1, default=float)
    # per-fill CSVs
    keep = ["cohort", "seed", "pair", "sig", "t", "sleeve", "half", "st", "entry_ms", "E"] + sum([[x, x + "_how", x + "_xms"] for x in EX], []) + \
           ["pk_before_stop", "pk12h", "ns_fix3", "ns_fix4", "ns_lock"]
    extra = {"a": ["rep_pct", "rep_pk", "rep_reason", "rep_E", "usd_per_pct"], "b": ["today_keep", "today_long", "today_wide", "bear", "old_fix3", "old_fix4", "old_lock"],
             "c": ["bear"]}
    F = pd.concat([Z[k][[c for c in keep + extra[k] if c in Z[k].columns]] for k in ("a", "b", "c")], ignore_index=True)
    F.to_csv(os.path.join(REP, "FRENZY_TP3_VS_TP4_TICKS_2026-10-08_fills.csv"), index=False)
    D.to_csv(os.path.join(REP, "FRENZY_TP3_VS_TP4_TICKS_2026-10-08_live.csv"), index=False)
    Wl.drop(columns=["_first"], errors="ignore").to_csv(os.path.join(REP, "FRENZY_TP3_VS_TP4_TICKS_2026-10-08_willy.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
