#!/usr/bin/env python3
"""FRENZY flat-base short (2026-10-22) — ANALYSIS + TABLES (pre-registration: reports/FRENZY_FLATBASE_SHORT_2026-10-22_PREREG.md).

Reads data/episodes_compact.pkl, data/signals.pkl, data/null.npy (study_flatbase_short_sim.py). Freezes the FLAT14 threshold from the
pre-2025-07 year episodes, labels every episode, and computes: every cell × cohort (flat / non-flat / unscored / all) per block, the
day-clustered bootstrap, the random-bar null (own + family-wise max-z), the shuffled-flat-label null (own + family-wise), the period
split, leave-one-month-out, best-5 % removal, concentration, dose-response, secondary metrics, survivorship, the live / master episode
list, the window test, and sizing at the $3,200 account. Writes:
  reports/FRENZY_FLATBASE_SHORT_2026-10-22_cells.csv      every cell × cohort × block
  reports/FRENZY_FLATBASE_SHORT_2026-10-22_trades.csv     every signal (one row) with the net % of all 54 exits (taken ones; blank = refused
                                                           by the one-open-short-per-pair rule)
  reports/FRENZY_FLATBASE_SHORT_2026-10-22_episodes.csv   every episode with its features and label
  reports/FRENZY_FLATBASE_SHORT_2026-10-22_live_master.csv every walk episode of blocks ①② with each entry's result at the primary exit
  reports/FRENZY_FLATBASE_SHORT_2026-10-22_tables.md      the printed tables
Usage: venv/bin/python scripts/study_flatbase_short_report.py [--perms 1000]
"""
import argparse, glob, io, json, os, pickle, sys, contextlib
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=1000); ap.add_argument("--freeze-only", action="store_true")
A = ap.parse_args()
rng = np.random.default_rng(20261022)
DL = os.path.expanduser("~/Downloads")
OUT = C.OUTP
EX = C.EXITS; NX = len(EX)
EXN = [f"{h}d|tp_{tp}|stop{int(s)}|{lv}x" for h, tp, s, lv in EX]
J_PRI = EX.index(C.PRIMARY[1:]); J_SEC = EX.index(C.SECONDARY[1:])
PRI = C.cell_name(*C.PRIMARY); SEC = C.cell_name(*C.SECONDARY)
ACCOUNT, RISK = 3200.0, 0.05
buf = io.StringIO()


def P(*a):
    print(*a); print(*a, file=buf)


def md(df, ff="{:+.2f}", index=False):
    d = df.reset_index() if index else df
    f = lambda v: (ff.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("–" if isinstance(v, (float, np.floating)) else str(v)))
    L = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    L += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(L)


# ───────────── episodes + frozen thresholds ─────────────
E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
E["key"] = list(zip(E.pair, E.on_t))
Y0 = E[(E.spike >= C.B_YEAR0) & (E.spike < C.FREEZE_CUT)]
THR = Y0.FLAT14.quantile(1 / 3)
QEDGE = Y0.FLAT14.quantile([0.2, 0.4, 0.6, 0.8]).values
SEC_THR = {m: Y0[m].quantile(1 / 3) for m in ("FLAT30", "RV14", "DEADVOL", "DIST90")}
P(f"## Frozen numbers (from {len(Y0)} year episodes with a spike 2025-01-01 → 2025-06-30; {Y0.FLAT14.notna().sum()} scored)")
P(f"- FLAT14 bottom-tertile threshold = {THR:.2f} % (14-d close range ≤ this = FLAT)")
P(f"- FLAT14 quintile edges = {', '.join(f'{x:.2f}' for x in QEDGE)} %")
P("- secondary bottom-tertile thresholds: " + " · ".join(f"{k} {v:.3f}" for k, v in SEC_THR.items()))
if A.freeze_only:
    sys.exit(0)
E["label"] = np.where(E.FLAT14.isna(), "unscored", np.where(E.FLAT14 <= THR, "flat", "nonflat"))
E["q14"] = np.where(E.FLAT14.isna(), -1, np.searchsorted(QEDGE, E.FLAT14.fillna(0), side="right"))
for m, v in SEC_THR.items():
    E[m + "_low"] = np.where(E[m].isna(), np.nan, (E[m] <= v).astype(float))
SYM = json.load(open(os.path.join(C.D, "symbols.json")))
TRADING = set(SYM["trading"])
OLDFIRST = {}
for d in C.W.SRC[:3]:
    for f in glob.glob(os.path.join(d, "*.csv")):
        try:
            t0 = int(pd.read_csv(f, usecols=[0], nrows=1).iloc[0, 0])
        except Exception:
            continue
        p_ = os.path.basename(f)[:-4]
        OLDFIRST[p_] = min(OLDFIRST.get(p_, 10**15), t0)
E["delisted"] = ~E.pair.isin(TRADING)
E["added_by_survivorship_fix"] = E.on_t < E.pair.map(OLDFIRST).fillna(10**15) + 11 * C.DAY
LAB = dict(zip(E.key, E.label))

# ───────────── real-episode evidence (blocks ① ②) ─────────────
def real_fills():
    m = pd.read_csv(os.path.join(C.R, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[m.entry_strategy.astype(str).str.startswith("FRENZY")][["pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage"]].assign(src="master")
    parts = []
    for f in sorted(glob.glob(os.path.join(DL, "scalpars_orders_*2026-10-*.csv"))):
        x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage"))
        x = x[x.entry_strategy.astype(str).str.startswith("FRENZY")]
        parts.append(x.assign(src="live" if "_live_" in f else "paper", f=os.path.basename(f)))
    o = pd.concat(parts).sort_values("f").drop_duplicates(["opened_at", "pair"], keep="last")
    a = pd.concat([m, o[["pair", "entry_strategy", "opened_at", "closed_at", "pnl_percentage", "src"]]])
    a["opened"] = pd.to_datetime(a.opened_at, format="mixed")
    return a.sort_values("src").drop_duplicates(["pair", "opened"], keep="first").reset_index(drop=True)


def journal():
    rows = []
    for f in sorted(glob.glob(os.path.join(DL, "scalpars_decisions_*2026-10-*.csv"))):
        try:
            x = pd.read_csv(f, usecols=["t", "pair", "gate", "strategy"], low_memory=False)
        except Exception:
            continue
        m = x.gate.astype(str).str.startswith("FRENZY") | x.strategy.astype(str).str.startswith("FRENZY")
        rows.append(x[m][["t", "pair"]])
    j = pd.concat(rows).drop_duplicates(); j["t"] = pd.to_datetime(j.t, format="mixed")
    return j


RF, JJ = real_fills(), journal()
E2 = E[E.spike >= pd.Timestamp("2026-10-01")].copy()
ev, nfill = [], []
for r in E2.itertuples():
    a = pd.to_datetime(r.on_t, unit="ms"); b = pd.to_datetime(r.off_t, unit="ms") + pd.Timedelta(minutes=5)
    jn = ((JJ.pair == r.pair) & (JJ.t >= a) & (JJ.t <= b)).sum(); fn = ((RF.pair == r.pair) & (RF.opened >= a) & (RF.opened <= b)).sum()
    ev.append(jn + fn > 0); nfill.append(fn)
E2["real_ep"] = ev; E2["real_fills"] = nfill
REAL = dict(zip(E2.key, E2.real_ep))
# parity: every real FRENZY fill inside a walk-flagged window
RF2 = RF[RF.opened >= pd.Timestamp("2026-10-02")]
inside = []
for r in RF2.itertuples():
    g = E[(E.pair == r.pair)]
    t = r.opened.value // 10**6
    inside.append(bool(((g.on_t <= t) & (g.off_t + C.BAR >= t)).any()))
P(f"\n- engine-cohort check: {sum(inside)} of {len(inside)} real FRENZY fills (master + live exports, ≥ 10-02) fall inside a walk-flagged window")

# ───────────── signals → long table ─────────────
S = pd.read_pickle(os.path.join(C.D, "signals.pkl"))
S["key"] = list(zip(S.pair, S.ep_on_t))
S = S.merge(E[["key", "spike", "FLAT14", "FLAT30", "RV14", "DEADVOL", "DIST90", "label", "q14", "delisted", "added_by_survivorship_fix",
               "FLAT30_low", "RV14_low", "DEADVOL_low", "DIST90_low"]], on="key", how="left")
S["entry_dt"] = pd.to_datetime(S.entry_ms, unit="ms")
S["block"] = np.where(S.entry_dt >= C.B_LIVE0, "1_live", np.where(S.entry_dt >= C.B_MAST0, "2_master", "3_year"))
S["real_ep"] = [REAL.get(k, False) for k in S.key]
S["day"] = S.entry_dt.dt.strftime("%Y-%m-%d"); S["month"] = S.entry_dt.dt.strftime("%Y-%m")
S["period"] = np.where(S.entry_dt < pd.Timestamp("2026-01-01"), "2025", np.where(S.entry_dt < pd.Timestamp("2026-07-01"), "2026H1", "2026H2"))
S = S.reset_index(drop=True)
G = np.stack(S.gross.values).astype(float); FU = np.stack(S.fund.values).astype(float); TK = np.stack(S.taken.values)
RS = np.stack(S.reason.values); XT = np.stack(S.exit_ms.values); MS = np.stack(S.fund_missing.values)
NET = G - C.FEES + FU; NOF = G - C.FEES
NET_T = np.where(TK, NET, np.nan)

# trades csv (wide)
TW = S[["pair", "entry", "spike", "entry_dt", "block", "real_ep", "label", "FLAT14", "FLAT30", "RV14", "DEADVOL", "DIST90", "close", "base",
        "peak", "vol24", "delisted", "data_open"]].copy()
for j, nm in enumerate(EXN):
    TW["net|" + nm] = np.round(NET_T[:, j], 3)
for j in (J_PRI, J_SEC):
    TW["nofund|" + EXN[j]] = np.round(np.where(TK[:, j], NOF[:, j], np.nan), 3)
    TW["fund|" + EXN[j]] = np.round(FU[:, j], 3)
    TW["reason|" + EXN[j]] = pd.Series(RS[:, j]).map({0: "time", 1: "stop", 2: "liq", 3: "tp", 4: "OPEN(mtm)", 5: "delist-settle"}).values
    TW["exit|" + EXN[j]] = pd.to_datetime(XT[:, j], unit="ms")
TW.sort_values("entry_dt").to_csv(OUT + "_trades.csv", index=False)
E.drop(columns=["key"]).to_csv(OUT + "_episodes.csv", index=False)


def boot(day, x, B=2000):
    d = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"])
    if len(d) < 3:
        return np.nan, np.nan
    s, c = d["sum"].values, d["count"].values; i = rng.integers(0, len(d), (B, len(d)))
    m = s[i].sum(1) / c[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summ(m, j, full=False):
    x = NET[m, j]; tk = TK[m, j]; x = x[tk]
    if not len(x):
        return dict(N=0)
    sub = S[m][tk]
    out = dict(N=len(x), days=sub.day.nunique(), pairs=sub.pair.nunique(), WR=(x > 0).mean() * 100, avg=x.mean(), med=np.median(x),
               nofund=NOF[m, j][tk].mean(), fund=FU[m, j][tk].mean(), tot=x.sum(), worst=x.min())
    lo, hi = boot(sub.day.values, x); out.update(lo=lo, hi=hi)
    if full:
        for p in ("2025", "2026H1", "2026H2"):
            out[p] = x[(sub.period == p).values].mean() if (sub.period == p).any() else np.nan
        lomo = [x[(sub.month != mo).values].mean() for mo in sub.month.unique()]
        out["lomo_min"] = min(lomo) if lomo else np.nan
        k = int(np.ceil(len(x) * 0.05)); out["ex_top5"] = np.sort(x)[:-k].mean() if len(x) >= 20 else np.nan
        pos = x.sum()
        pc = pd.Series(x).groupby(sub.pair.values).sum(); dc = pd.Series(x).groupby(sub.day.values).sum()
        out["top_pair"] = pc.idxmax(); out["top_pair_sh"] = pc.max() / pos * 100 if pos > 0 else np.nan
        out["top_day"] = dc.idxmax(); out["top_day_sh"] = dc.max() / pos * 100 if pos > 0 else np.nan
        out["worst_day"] = dc.idxmin(); out["worst_day_sum"] = dc.min()
        out["stop_liq%"] = np.isin(RS[m, j][tk], [1, 2]).mean() * 100
        out["fund_missing%"] = MS[m, j][tk].mean() * 100
    return out


# ───────────── all cells × cohort × block ─────────────
rows = []
for en in C.ENTRIES:
    me = (S.entry == en).values
    for blk in ("3_year", "2_master", "1_live"):
        mb = me & (S.block == blk).values & ((S.real_ep.values) if blk != "3_year" else True)
        for coh in ("flat", "nonflat", "unscored", "all"):
            mc = mb & ((S.label == coh).values if coh != "all" else True)
            for j, nm in enumerate(EXN):
                s = summ(mc, j, full=(blk == "3_year" and coh in ("flat", "nonflat")))
                s.update(cell=f"{en}|{nm}", entry=en, exit=nm, block=blk, cohort=coh); rows.append(s)
CELLS = pd.DataFrame(rows)

# ───────────── nulls (year block) ─────────────
NULL = np.load(os.path.join(C.D, "null.npy"), mmap_mode="r")
NID = pickle.load(open(os.path.join(C.D, "null_eps.pkl"), "rb"))
nrow = {k: i for i, k in enumerate(NID)}
YR = (S.block == "3_year").values
# (a) random bars in the same FLAT episodes
fw_rows = []
NM = {}
for en in C.ENTRIES:
    m = YR & (S.entry == en).values & (S.label == "flat").values
    for j, nm in enumerate(EXN):
        mm = m & TK[:, j]
        ks = [nrow[k] for k in S.key[mm] if k in nrow]
        if len(ks) < 5:
            continue
        ks = np.array(sorted(ks))
        nmean = NULL[ks, :, j].astype(float).mean(0)       # 200 shuffles
        NM[f"{en}|{nm}"] = (NET[mm, j].mean(), nmean)
cells_n = list(NM)
real = np.array([NM[c][0] for c in cells_n]); nul = np.stack([NM[c][1] for c in cells_n], 1)   # B × cells
mu, sd = nul.mean(0), nul.std(0, ddof=1)
z = (real - mu) / sd; zmax = ((nul - mu) / sd).max(1)
RB = pd.DataFrame({"cell": cells_n, "real": real, "rand_mean": mu, "p_rand_own": (nul >= real).mean(0), "p_rand_fw": [(zmax >= x).mean() for x in z]})
# (b)+(c) flat vs non-flat, shuffled labels (episode-level permutation over all scored year episodes)
SC = S[YR & S.label.isin(["flat", "nonflat"]).values]
ek = sorted(set(SC.key)); epos = {k: i for i, k in enumerate(ek)}
lab = np.array([LAB[k] == "flat" for k in ek])
PERM = np.stack([rng.permutation(lab) for _ in range(A.perms)])             # perms × episodes
dl_rows, DN = [], {}
for en in C.ENTRIES:
    m = YR & (S.entry == en).values & S.label.isin(["flat", "nonflat"]).values
    for j, nm in enumerate(EXN):
        mm = m & TK[:, j]
        if mm.sum() < 20:
            continue
        idx = np.array([epos[k] for k in S.key[mm]]); x = NET[mm, j]
        f = lab[idx]
        d_real = x[f].mean() - x[~f].mean()
        Pm = PERM[:, idx].astype(float)
        nf = Pm.sum(1); d_null = (Pm @ x) / nf - ((1 - Pm) @ x) / (len(x) - nf)
        DN[f"{en}|{nm}"] = (d_real, d_null)
cells_d = list(DN)
dr = np.array([DN[c][0] for c in cells_d]); dn = np.stack([DN[c][1] for c in cells_d], 1)
mu2, sd2 = dn.mean(0), dn.std(0, ddof=1); z2 = (dr - mu2) / sd2; zmax2 = ((dn - mu2) / sd2).max(1)
DL_ = pd.DataFrame({"cell": cells_d, "delta_flat_minus_nonflat": dr, "p_label_own": (dn >= dr).mean(0), "p_label_fw": [(zmax2 >= x).mean() for x in z2]})
CY = CELLS[(CELLS.block == "3_year")]
CF = CY[CY.cohort == "flat"].merge(RB, on="cell", how="left").merge(DL_, on="cell", how="left")
CF = CF.merge(CY[CY.cohort == "nonflat"][["cell", "N", "avg", "lo", "hi"]].rename(columns={"N": "N_nonflat", "avg": "avg_nonflat", "lo": "lo_nonflat", "hi": "hi_nonflat"}), on="cell")
CELLS = CELLS.merge(RB, on="cell", how="left").merge(DL_, on="cell", how="left")
CELLS.loc[~((CELLS.block == "3_year") & (CELLS.cohort == "flat")), ["real", "rand_mean", "p_rand_own", "p_rand_fw"]] = np.nan
CELLS.loc[~((CELLS.block == "3_year") & (CELLS.cohort == "flat")), ["delta_flat_minus_nonflat", "p_label_own", "p_label_fw"]] = np.nan
CELLS.to_csv(OUT + "_cells.csv", index=False)

# ───────────── print: headline ─────────────
P(f"\n## Episodes\n- year-block episodes (spike 2025-01-01 → 2026-10-01): {((E.spike >= C.B_YEAR0) & (E.spike < C.B_YEAR1)).sum()} · flat "
  f"{((E.spike >= C.B_YEAR0) & (E.spike < C.B_YEAR1) & (E.label == 'flat')).sum()} · non-flat {((E.spike >= C.B_YEAR0) & (E.spike < C.B_YEAR1) & (E.label == 'nonflat')).sum()}"
  f" · unscored {((E.spike >= C.B_YEAR0) & (E.spike < C.B_YEAR1) & (E.label == 'unscored')).sum()}")
P(f"- signals: {len(S)} ({S.entry.value_counts().to_dict()})")
P("\n## Year block, FLAT cohort — all 4 entries at the pre-registered primary exit (7 d · TP base · stop 25 · 1×) and the 216-cell ranking")
cols = ["cell", "N", "days", "WR", "avg", "lo", "hi", "nofund", "fund", "2025", "2026H1", "2026H2", "lomo_min", "ex_top5", "top_pair_sh",
        "top_day_sh", "avg_nonflat", "delta_flat_minus_nonflat", "p_label_own", "p_label_fw", "rand_mean", "p_rand_own", "p_rand_fw"]
P(md(CF[CF.exit == EXN[J_PRI]][cols]))
P("\nTop 15 cells by the day-bootstrap lower bound (flat, year):")
P(md(CF.sort_values("lo", ascending=False).head(15)[cols]))
P(f"\nCells with lo > 0: {(CF.lo > 0).sum()} of {len(CF)} · mean > 0: {(CF.avg > 0).sum()} · min p_label_fw {CF.p_label_fw.min():.3f} · "
  f"min p_rand_fw {CF.p_rand_fw.min():.3f}")
P("\nNon-flat cohort, primary exit, all entries (year):")
CN = CY[CY.cohort == "nonflat"]
P(md(CN[CN.exit == EXN[J_PRI]][["cell", "N", "days", "WR", "avg", "lo", "hi", "nofund", "fund", "2025", "2026H1", "2026H2"]]))
P("\nAll cohorts, every entry × hold (TP base · stop 25 · 1×), year — avg % [lo, hi] N:")
tb = []
for en in C.ENTRIES:
    for hd in C.HOLDS_D:
        nm = f"{hd}d|tp_base|stop25|1x"
        r = {"entry": en, "hold": f"{hd}d"}
        for coh in ("flat", "nonflat", "unscored"):
            x = CY[(CY.cell == f"{en}|{nm}") & (CY.cohort == coh)].iloc[0]
            r[coh] = f"{x.avg:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] {int(x.N)}" if x.N else "–"
        tb.append(r)
P(md(pd.DataFrame(tb)))
P("\nAll cohorts, every entry × hold, NO stop · no TP · 1× (the operator's 'just hold it' short), year:")
tb = []
for en in C.ENTRIES:
    for hd in C.HOLDS_D:
        nm = f"{hd}d|tp_none|stop0|1x"
        r = {"entry": en, "hold": f"{hd}d"}
        for coh in ("flat", "nonflat"):
            x = CY[(CY.cell == f"{en}|{nm}") & (CY.cohort == coh)].iloc[0]
            r[coh] = f"{x.avg:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] {int(x.N)} · WR {x.WR:.0f}% · worst {x.worst:+.0f} · pre-fund {x.nofund:+.2f}" if x.N else "–"
        tb.append(r)
P(md(pd.DataFrame(tb)))

# ───────────── primary: three blocks, robustness ─────────────
def blocks(en, j, coh="flat"):
    out = []
    for blk, nmb in (("1_live", "① live"), ("2_master", "② master"), ("3_year", "③ year")):
        m = (S.entry == en).values & (S.block == blk).values & ((S.label == coh).values if coh != "all" else True)
        if blk != "3_year":
            m &= S.real_ep.values
        s = summ(m, j)
        nopen = int((RS[m, j][TK[m, j]] == 4).sum()) if s.get("N") else 0
        out.append(dict(block=nmb, N=s.get("N", 0), days=s.get("days", 0), WR=s.get("WR", np.nan), avg=s.get("avg", np.nan), lo=s.get("lo", np.nan),
                        hi=s.get("hi", np.nan), nofund=s.get("nofund", np.nan), fund=s.get("fund", np.nan), tot=s.get("tot", np.nan),
                        worst=s.get("worst", np.nan), open_mtm=nopen))
    return pd.DataFrame(out)


for en, hd in (("S2", 7), ("S3", 7), ("S1", 7), ("S4", 7)):
    j = EX.index((hd, "base", 25.0, 1))
    for coh in ("flat", "nonflat"):
        P(f"\n### {en}|{EXN[j]} — {coh} — blocks ①②③ (①② = real episodes; open trades marked to market)")
        P(md(blocks(en, j, coh)))
pri = CF[CF.cell == PRI].iloc[0]
P(f"\n### PRIMARY {PRI} robustness (flat, year): " + " · ".join(f"{k} {pri[k]:+.2f}" if isinstance(pri[k], (float, np.floating)) else f"{k} {pri[k]}"
                                                                 for k in ["N", "days", "avg", "lo", "hi", "2025", "2026H1", "2026H2", "lomo_min", "ex_top5",
                                                                           "top_pair", "top_pair_sh", "top_day", "top_day_sh", "worst_day", "worst_day_sum",
                                                                           "stop_liq%", "fund_missing%", "nofund", "fund"]))
mp_ = YR & (S.entry == "S2").values & (S.label == "flat").values & TK[:, J_PRI]
x = NET[mp_, J_PRI]; sub = S[mp_]
mo = pd.DataFrame({"m": sub.month.values, "x": x}).groupby("m").x.agg(["count", "mean", "sum"])
P("\nPrimary by month (flat, year): " + " · ".join(f"{i} {r['count']:.0f}/{r['mean']:+.2f}" for i, r in mo.iterrows()))

# ───────────── dose-response ─────────────
P("\n## Dose-response — FLAT14 quintiles (edges frozen pre-2025-07), year, TP base · stop 25 · 1×, 7 d")
dr_rows = []
for en in C.ENTRIES:
    for q in range(5):
        m = YR & (S.entry == en).values & (S.q14 == q).values
        s = summ(m, J_PRI)
        dr_rows.append(dict(entry=en, quintile=f"Q{q + 1}", range=f"{([0] + list(QEDGE))[q]:.0f}–{(list(QEDGE) + [np.inf])[q]:.0f} %", N=s.get("N", 0),
                            WR=s.get("WR", np.nan), avg=s.get("avg", np.nan), lo=s.get("lo", np.nan), hi=s.get("hi", np.nan), nofund=s.get("nofund", np.nan),
                            fund=s.get("fund", np.nan)))
P(md(pd.DataFrame(dr_rows)))
P("\nSame, no stop · no TP · 1× · 14 d (pure 'hold it two weeks'):")
j14 = EX.index((14, "none", 0.0, 1))
dr_rows = []
for en in C.ENTRIES:
    for q in range(5):
        m = YR & (S.entry == en).values & (S.q14 == q).values
        s = summ(m, j14)
        dr_rows.append(dict(entry=en, quintile=f"Q{q + 1}", N=s.get("N", 0), WR=s.get("WR", np.nan), avg=s.get("avg", np.nan), lo=s.get("lo", np.nan),
                            hi=s.get("hi", np.nan), nofund=s.get("nofund", np.nan), fund=s.get("fund", np.nan), worst=s.get("worst", np.nan)))
P(md(pd.DataFrame(dr_rows)))

# ───────────── secondary metrics ─────────────
P("\n## Secondary flatness metrics (bottom tertile vs rest; year; primary exit) — reported, never a ship basis")
sr = []
for met in ("FLAT30", "RV14", "DEADVOL", "DIST90"):
    for en in ("S2", "S3"):
        for v, nmv in ((1.0, "low tertile"), (0.0, "rest")):
            m = YR & (S.entry == en).values & (S[met + "_low"] == v).values
            s = summ(m, J_PRI)
            sr.append(dict(metric=met, entry=en, group=nmv, N=s.get("N", 0), WR=s.get("WR", np.nan), avg=s.get("avg", np.nan), lo=s.get("lo", np.nan),
                           hi=s.get("hi", np.nan)))
P(md(pd.DataFrame(sr)))

# ───────────── funding ─────────────
P("\n## Funding (the short receives + / pays −), % of notional per trade, year, primary-family exits")
fr = []
for coh in ("flat", "nonflat"):
    for hd in C.HOLDS_D:
        j = EX.index((hd, "none", 0.0, 1))
        m = YR & (S.label == coh).values & TK[:, j]
        f = FU[m, j]
        fr.append(dict(cohort=coh, hold=f"{hd}d (held to the end)", N=len(f), mean=f.mean(), median=np.median(f), p5=np.percentile(f, 5),
                       share_paying=(f < 0).mean() * 100, missing=MS[m, j].mean() * 100))
P(md(pd.DataFrame(fr)))

# ───────────── survivorship ─────────────
P("\n## Survivorship")
YE = E[(E.spike >= C.B_YEAR0) & (E.spike < C.B_YEAR1)]
P(f"- year episodes on pairs no longer trading (delisted / settling): {YE.delisted.sum()} of {len(YE)} ({YE.delisted.mean() * 100:.1f} %), "
  f"{YE[YE.delisted].pair.nunique()} of {YE.pair.nunique()} pairs · flat episodes delisted {YE[YE.label == 'flat'].delisted.mean() * 100:.1f} % vs non-flat "
  f"{YE[YE.label == 'nonflat'].delisted.mean() * 100:.1f} %")
P(f"- episodes ADDED by fetching the delisted pairs' history (absent from the 10-20 study's data): {YE.added_by_survivorship_fix.sum()}")
sv = []
for coh in ("flat", "nonflat"):
    for dl, nmv in ((True, "delisted pairs"), (False, "still trading")):
        m = YR & (S.entry == "S2").values & (S.label == coh).values & (S.delisted == dl).values
        s = summ(m, J_PRI)
        sv.append(dict(cohort=coh, pairs=nmv, N=s.get("N", 0), WR=s.get("WR", np.nan), avg=s.get("avg", np.nan), lo=s.get("lo", np.nan), hi=s.get("hi", np.nan)))
    m = YR & (S.entry == "S2").values & (S.label == coh).values & ~S.added_by_survivorship_fix.values
    s = summ(m, J_PRI)
    sv.append(dict(cohort=coh, pairs="10-20 study's universe only", N=s.get("N", 0), WR=s.get("WR", np.nan), avg=s.get("avg", np.nan), lo=s.get("lo", np.nan), hi=s.get("hi", np.nan)))
P(md(pd.DataFrame(sv)))

# ───────────── live / master episodes ─────────────
P("\n## Blocks ① ② — every walk episode with a spike ≥ 2026-10-01 (real = bot FRENZY evidence); primary exit per entry; OPEN = marked to market")
LM = []
KIDX = {}
for k_, kk in enumerate(S.key):
    KIDX.setdefault(kk, []).append(k_)
for r in E2.sort_values("spike").itertuples():
    row = dict(pair=r.pair, spike=str(r.spike)[:16], real=bool(r.real_ep), real_fills=int(r.real_fills), FLAT14=r.FLAT14, label=r.label)
    for en in C.ENTRIES:
        ks_ = [k_ for k_ in KIDX.get(r.key, []) if S.entry[k_] == en]
        if ks_:
            k = ks_[0]
            tag = {0: "", 1: " stop", 2: " LIQ", 3: " tp", 4: " OPEN", 5: " settle"}[int(RS[k, J_PRI])]
            row[en] = (f"{NET[k, J_PRI]:+.1f}{tag}" if TK[k, J_PRI] else "busy") + f" @{str(S.entry_dt[k])[5:16]}"
            row[en + "_hold14_nostop"] = f"{NET[k, j14]:+.1f}" + {0: "", 1: " stop", 2: " LIQ", 3: " tp", 4: " OPEN", 5: " settle"}[int(RS[k, j14])]
        else:
            row[en] = ""; row[en + "_hold14_nostop"] = ""
    LM.append(row)
LM = pd.DataFrame(LM)
LM.to_csv(OUT + "_live_master.csv", index=False)
P(md(LM[LM.real], ff="{:.1f}"))
P(f"\n(walk episodes ≥ 10-01: {len(LM)}, real {LM.real.sum()}; flat {(LM.label == 'flat').sum()}, real flat {((LM.label == 'flat') & LM.real).sum()})")
P("\nIOST episodes (any block):")
P(md(E[E.pair == "IOSTUSDT"][["pair", "spike", "FLAT14", "FLAT30", "DEADVOL", "DIST90", "label", "base"]], ff="{:.4g}"))
io_ = S[S.pair == "IOSTUSDT"]
for k in io_.index:
    P(f"IOST {S.entry[k]} entry {S.entry_dt[k]} close {S.close[k]:.6g} base {S.base[k]:.6g} peak {S.peak[k]:.6g} · primary {NET[k, J_PRI]:+.2f} "
      f"({['time', 'stop', 'liq', 'tp', 'OPEN', 'settle'][RS[k, J_PRI]]}, funding {FU[k, J_PRI]:+.2f}) · 14 d no stop {NET[k, j14]:+.2f} "
      f"({['time', 'stop', 'liq', 'tp', 'OPEN', 'settle'][RS[k, j14]]}, funding {FU[k, j14]:+.2f})")

# window test (bar 5)
m = (S.entry == "S2").values & S.block.isin(["1_live", "2_master"]).values & S.real_ep.values & (S.label == "flat").values & TK[:, J_PRI]
wv = pd.DataFrame({"d": S.day[m], "x": NET[m, J_PRI]}).groupby("d").x.mean()
P(f"\n- WINDOW TEST (bar 5): primary on real flat ①② episodes → {len(wv)} windows, mean of window means {wv.mean() if len(wv) else np.nan:+.2f}, "
  f"positive {(wv > 0).sum()} of {len(wv)}: " + ", ".join(f"{d} {v:+.1f}" for d, v in wv.items()))

# ───────────── sizing / ruin ─────────────
def sizing(en, j, label="flat", maxc=5):
    m = YR & (S.entry == en).values & (S.label == label).values & TK[:, j]
    idx = np.nonzero(m)[0]
    x = NET[idx, j]
    p1 = np.percentile(x, 1)
    notional = ACCOUNT * RISK / abs(p1) * 100
    order = idx[np.argsort(S.entry_ms.values[idx])]
    openx, taken = [], []
    for k in order:
        openx = [(xt, pr) for xt, pr in openx if xt > S.entry_ms[k]]
        if len(openx) >= maxc or any(pr == S.pair[k] for _, pr in openx):
            continue
        openx.append((XT[k, j], S.pair[k])); taken.append(k)
    taken = np.array(taken)
    pnl = NET[taken, j] / 100 * notional
    fund_usd = FU[taken, j] / 100 * notional
    ex = pd.to_datetime(XT[taken, j], unit="ms").normalize()
    daily_real = pd.Series(pnl, index=ex).groupby(level=0).sum()
    days = pd.date_range(S.entry_dt[taken].min().normalize(), max(ex.max(), S.entry_dt[taken].max().normalize()), freq="D")
    # daily mark-to-market: realised + open marks (raw short mark at the day index, until exit)
    eq = []
    MTM = np.stack(S.mtm.values[taken]).astype(float)
    ent = S.entry_ms.values[taken]; xt = XT[taken, j]
    for d in days:
        de = d.value // 10**6 + C.DAY
        closed = xt <= de
        r_ = pnl[closed].sum()
        op = ~closed & (ent <= de)
        k = np.clip(((de - ent[op]) // C.DAY).astype(int) - 1, 0, 13)
        mk = MTM[op][np.arange(op.sum()), k]
        eq.append(r_ + np.nansum(mk / 100 * notional))
    eq = pd.Series(eq, index=days)
    dd = (eq - eq.cummax()).min()
    cr = daily_real.cumsum(); ddr = (cr - cr.cummax()).min()
    return dict(cell=f"{en}|{EXN[j]}", p1=p1, notional=notional, N_taken=len(taken), N_signals=len(idx), sum=pnl.sum(), avg_usd=pnl.mean(),
                worst_trade=pnl.min(), worst_day=daily_real.min(), maxdd_realised=ddr, maxdd_mtm=dd, maxdd_pct=dd / ACCOUNT * 100,
                funding_usd=fund_usd.sum(), funding_per_trade=fund_usd.mean())


P("\n## Sizing at $3,200 (1st-percentile loss = 5 % = $160; ≤ 5 concurrent shorts, one per pair; year flat cohort)")
szr = [sizing("S2", J_PRI), sizing("S3", J_SEC)]
best = CF.sort_values("lo", ascending=False).iloc[0]
be, bx = best.cell.split("|", 1)
szr.append(sizing(be, EXN.index(bx)))
szr.append(sizing("S2", j14))
P(md(pd.DataFrame(szr)))

open(OUT + "_tables.md", "w").write(buf.getvalue())
print("written", OUT + "_tables.md")
