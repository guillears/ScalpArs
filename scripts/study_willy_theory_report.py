#!/usr/bin/env python3
"""WILLY THEORY — step 5: every pre-registered table (W0–W8 + W6b) in three blocks, the gates, the shuffle null.
Writes reports/willy_theory/tables.md, gates.csv, ms_rules.csv. Reads bt.pkl / alt.pkl / ms.pkl (study_willy_theory_data.py).
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402

OUT = []
ERAS = ("2025", "2026-H1", "2026-H2")


def P(*a):
    OUT.append(" ".join(str(x) for x in a))


def f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def boot_neg(y, day, n=4000, seed=7):
    """P(mean < 0) by a day-clustered bootstrap."""
    p, lo, hi = W.boot_mean(y, day, n=n, seed=seed)
    return (1 - p) if np.isfinite(p) else np.nan


B = pd.read_pickle(os.path.join(W.OUT, "bt.pkl"))
A = pd.read_pickle(os.path.join(W.OUT, "alt.pkl"))
MS = pd.read_pickle(os.path.join(W.OUT, "ms.pkl"))
B["year"] = np.where(B.era == "2025", "2025", "2026")
B["t50"] = B.top50.astype(float)
BT = B[B.tick].copy()                      # verdict rows (RH, 90-min window fully tick-covered)
BE = W.be_wr(BT.y)
NOBS_BOOT = 4000


# ───────────────────────── generic block-cell evaluation ─────────────────────────
def cell_eval(D, mask, ycol="y", nboot=NOBS_BOOT):
    """blocked = D[mask] → dict of gate stats (on D's rows)."""
    m = np.asarray(mask, bool) & np.isfinite(D[ycol].values)
    y = D[ycol].values; day = D.day.values
    ok = np.isfinite(y)
    yb = y[m]
    out = dict(N=int(m.sum()), N_all=int(ok.sum()), delta=0.0, p_delta=np.nan)
    if out["N"] == 0:
        return out
    out["wr"] = (yb > 0).mean() * 100
    out["avg"] = yb.mean()
    out["avg_kept"] = y[ok & ~m].mean() if (ok & ~m).any() else np.nan
    out["sum"] = yb.sum()
    out["days"] = int(pd.Series(day[m]).nunique())
    out["p_neg"] = boot_neg(yb, day[m], n=nboot)
    s = pd.Series(yb).groupby(day[m]).sum()
    lp = pd.Series(yb).groupby(D.pair.values[m]).sum()
    tot = yb.sum()
    if tot < 0:
        out["max_day_share"] = float(s.min() / tot) if s.min() < 0 else 0.0
        out["max_pair_share"] = float(lp.min() / tot) if lp.min() < 0 else 0.0
    else:
        out["max_day_share"] = out["max_pair_share"] = np.nan
    v = np.where(m, -y, 0.0)[ok]
    out["delta"] = v.mean()                                   # refuse Δ per signal of the cohort
    pp, lo, hi = W.boot_mean(v, day[ok], n=nboot)
    out["p_delta"] = pp
    return out


def gate(row):
    """locked filter gate on the pooled tick cohort + walk-forward."""
    c1 = row["wr"] < BE
    c2 = row["p_neg"] >= 0.95
    c3 = row["days"] >= 8 and (not np.isfinite(row["max_day_share"]) or row["max_day_share"] < 0.5) and \
        (not np.isfinite(row["max_pair_share"]) or row["max_pair_share"] < 0.5)
    c4 = row["N"] >= 15
    wf = row.get("d_2025", np.nan) > 0 and row.get("d_2026", np.nan) > 0 and row.get("d_match", np.nan) > 0 and row.get("d_r1", np.nan) > 0
    return dict(g1_wr=c1, g2_conf=c2, g3_conc=c3, g4_n=c4, wf=bool(wf), PASS=bool(c1 and c2 and c3 and c4 and wf))


BLOCKS = {}            # name → (BT mask function, MS mask function or None, theory text)


def reg(name, fn, txt, ms_fn=None):
    BLOCKS[name] = (fn, ms_fn if ms_fn is not None else fn, txt)


reg("W1_DEEP", lambda d: d.ent_bar_ret <= -2.0, "deep red bar (≤ −2 %) = distribution, not a pause")
reg("W1_SHALLOW", lambda d: d.ent_bar_ret > -0.75, "shallow red bar (> −0.75 %) = no real pullback, buying high")
reg("W2_HEAVY", lambda d: d.red_vs_spike_q >= 1.0, "red-bar volume ≥ the spike bar's = heavy selling")
reg("W2_SELLERS", lambda d: d.tbs_red < 45, "red-bar taker-buy share < 45 % = aggressive sellers")
reg("W2_LAST1M_SELL", lambda d: d.tbs_red_last1m < 40, "last minute of the red bar < 40 % taker buys = selling into the close")
reg("W3_ONECANDLE", lambda d: d.spike_conc >= 0.6, "one-candle spike = news / squeeze, no base of buyers")
reg("W3_STAIR", lambda d: d.spike_conc <= 0.4, "staircase spike (read both ways)")
reg("W3_WICK", lambda d: d.spike_uwick >= 0.40, "spike-bar upper wick ≥ 40 % = rejection at the top")
reg("W3_BIG", lambda d: d.spike_ret6 >= 10, "big spike (≥ 10 % in 30 min) = overextended")
reg("W3_VOLX", lambda d: d.spike_volx >= 40, "spike volume ≥ 40× normal = blow-off")
reg("W3_FARHIGH", lambda d: d.off_high <= -5, "entry ≥ 5 % below the spike high = the unwind already started")
reg("W4_WAIT4", lambda d: d.wait_bars >= 4, "first red ≥ 20 min after arming = momentum faded")
reg("W4_LATEFLAG", lambda d: d.min_from_spike > 60, "trigger far from the spike (late-verified flag)")
reg("W5_BELOWVWAP", lambda d: d.vs_vwap <= 0, "entry at/below the spike VWAP = buyers lost the average")
reg("W5_RETRACE", lambda d: d.retrace_frac >= 0.35, "≥ 35 % of the pump already given back")
reg("W5_SMALLKEEP", lambda d: d.vs_base_pct < 8, "entry < 8 % above the pre-spike base")
reg("W6_BTC1H_SLOPE", lambda d: d.btc_1h_slope <= 0, "BTC 1h EMA20 slope ≤ 0 (market not supporting)")
reg("W6_BTC15", lambda d: d.btc_15m <= 0, "BTC 15-min return ≤ 0")
reg("W6_BTC1H", lambda d: d.btc_1h <= 0, "BTC 1-hour return ≤ 0")
reg("W6_BREADTH", lambda d: d.brd_1h < 50, "< 50 % of the top-50 up over 1 h")
reg("W6B_BETA", lambda d: d.btc_bar < -0.02, "BTC red on the same 5m bar = beta pullback")
reg("W6B_IDIO", lambda d: d.btc_bar > 0.02, "BTC green on the same bar = coin-specific pullback")
reg("W6B_MKTRED", lambda d: d.brd_red >= 60, "≥ 60 % of the top-50 red on the same bar")
reg("W6B_MKTSELL", lambda d: d.mkt_tbs < 48, "market (BTC+ETH+SOL) taker-buy share < 48 % in the bar")
reg("W7_HIGHTURN", lambda d: d.R_ent >= 0.5, "turnover ≥ 0.5 (within the live < 1)")
reg("W7_SMALLCAP", lambda d: d.mcap < 100e6, "market cap < $100M")
reg("W7_NOTTOP50", lambda d: ~d.top50.astype(bool), "not in the previous day's top-50")
reg("W7_NEWLIST", lambda d: d.listing_days < 180, "listed < 180 days")


def eval_all(Bv, Bfull, nboot=NOBS_BOOT):
    rows = []
    for name, (fn, _, txt) in BLOCKS.items():
        m = fn(Bv).fillna(False).values if hasattr(fn(Bv), "fillna") else fn(Bv)
        sc = Bv.eval if False else None
        r = dict(cell=name, theory=txt, **cell_eval(Bv, m, nboot=nboot))
        r["scored"] = int(np.isfinite(_score(Bv, name)).sum())
        for y in ("2025", "2026"):
            s = Bv[Bv.year == y]
            r[f"d_{y}"] = cell_eval(s, fn(s).fillna(False).values, nboot=200 if nboot < NOBS_BOOT else 1000)["delta"] if len(s) else np.nan
        for e in ERAS:
            s = Bv[Bv.era == e]
            ce = cell_eval(s, fn(s).fillna(False).values, nboot=200 if nboot < NOBS_BOOT else 1000)
            r[f"avg_{e}"] = ce.get("avg", np.nan); r[f"n_{e}"] = ce.get("N", 0); r[f"d_{e}"] = ce.get("delta", np.nan)
            r[f"pd_{e}"] = ce.get("p_delta", np.nan); r[f"wr_{e}"] = ce.get("wr", np.nan)
        s = Bv[Bv.MATCH]
        r["d_match"] = cell_eval(s, fn(s).fillna(False).values, nboot=200)["delta"] if len(s) else np.nan
        # R1 refute ruler on ALL resolved rows (1m)
        s = Bfull[np.isfinite(Bfull.y1)]
        r["d_r1"] = cell_eval(s, fn(s).fillna(False).values, ycol="y1", nboot=200)["delta"]
        if r["N"]:
            r.update(gate(r))
        rows.append(r)
    return pd.DataFrame(rows)


SCORE_COL = {"W1": "ent_bar_ret", "W2_HEAVY": "red_vs_spike_q", "W2_SELLERS": "tbs_red", "W2_LAST1M_SELL": "tbs_red_last1m",
             "W3_ONECANDLE": "spike_conc", "W3_STAIR": "spike_conc", "W3_WICK": "spike_uwick", "W3_BIG": "spike_ret6", "W3_VOLX": "spike_volx",
             "W3_FARHIGH": "off_high", "W4": "wait_bars", "W4_LATEFLAG": "min_from_spike", "W5_BELOWVWAP": "vs_vwap", "W5_RETRACE": "retrace_frac",
             "W5_SMALLKEEP": "vs_base_pct", "W6_BTC1H_SLOPE": "btc_1h_slope", "W6_BTC15": "btc_15m", "W6_BTC1H": "btc_1h", "W6_BREADTH": "brd_1h",
             "W6B_BETA": "btc_bar", "W6B_IDIO": "btc_bar", "W6B_MKTRED": "brd_red", "W6B_MKTSELL": "mkt_tbs", "W7_HIGHTURN": "R_ent",
             "W7_SMALLCAP": "mcap", "W7_NOTTOP50": "R_ent", "W7_NEWLIST": "listing_days"}


def _score(D, name):
    col = SCORE_COL.get(name) or SCORE_COL.get(name[:2]) or SCORE_COL.get(name.split("_")[0])
    return D[col].astype(float).values if col in D else np.full(len(D), np.nan)


def bucket_table(D, col, edges, labels, title, ycol="y"):
    P(f"\n**{title}** (`{col}`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))\n")
    P("| bucket | " + " | ".join(f"{e} N · WR · avg" for e in ERAS) + " | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |")
    P("|---|" + "---|" * (len(ERAS) + 4))
    x = D[col].astype(float)
    b = pd.cut(x, edges, labels=labels, right=True)
    BF = B.copy(); bf = pd.cut(BF[col].astype(float), edges, labels=labels, right=True)
    for lab in labels:
        s = D[b == lab]
        cells = []
        for e in ERAS:
            q = s[s.era == e][ycol]
            cells.append(f"{len(q)} · {(q > 0).mean() * 100:.0f}% · {f(q.mean())}" if len(q) else "0")
        q = s[ycol]
        pp = W.boot_mean(q.values, s.day.values, n=1000)[0] if len(q) > 3 else np.nan
        mm = s[s.MATCH][ycol].mean() if s.MATCH.any() else np.nan
        r1 = BF[(bf == lab)].y1.mean()
        P(f"| {lab} | " + " | ".join(cells) + f" | {len(q)} · {(q > 0).mean() * 100:.0f}% · {f(q.mean())} | {f(pp, 2, False)} | {f(mm)} | {f(r1)} |")
    nn = x.isna().sum()
    P(f"\nunscored rows (never counted on either side): {nn}")


def ms_rule_table(name):
    """master + live: before / after per trade in % and $ for the BLOCK cell."""
    fn = BLOCKS[name][1]
    try:
        m = fn(MS).fillna(False).values
        sc = _score(MS, name)
    except Exception:  # noqa: BLE001
        m = np.zeros(len(MS), bool); sc = np.full(len(MS), np.nan)
    rows = []
    for (blk, bt), g in MS.groupby(["block", "batch"], sort=False):
        idx = g.index
        mm = pd.Series(m, index=MS.index)[idx]
        sc_ok = np.isfinite(pd.Series(sc, index=MS.index)[idx])
        rows.append(dict(cell=name, block=blk, batch=bt, n=len(g), scored=int(sc_ok.sum()), blocked=int(mm.sum()),
                         pct_traded=g.pct_traded.sum(), usd_traded=g.usd_traded.sum(),
                         pct_traded_rule=g.pct_traded[~mm].sum(), usd_traded_rule=g.usd_traded[~mm].sum(),
                         pct_today=g.y.sum(), usd_today=g.usd_today.sum(), pct_today_rule=g.y[~mm].sum(), usd_today_rule=g.usd_today[~mm].sum(),
                         blocked_list=", ".join(f"{p} {y:+.2f}" for p, y in zip(g.pair[mm], g.pct_traded[mm]))))
    return pd.DataFrame(rows)


# ═════════════════════════ REPORT ═════════════════════════
P("# WILLY THEORY — tables (generated by scripts/study_willy_theory_report.py)\n")
P(f"BT rebuilt WILLY entries {len(B):,} · priced (resolved RH) {B.why.isin(['TP', 'LATE_TP', 'STOP', 'CAP']).sum():,} · **tick-verdict rows {len(BT):,}** · "
  f"MATCH (bar-exact replay fills, seed 1) {B.MATCH.sum():,} (tick {BT.MATCH.sum():,}) · R1-resolved {np.isfinite(B.y1).sum():,}")
P(f"Entry-source: {B.esrc.value_counts().to_dict()} · path src {B.src.value_counts().to_dict()}")
P(f"Sleeve breakeven WR (RH ticks, pooled) **{BE:.1f} %**; per era " + " · ".join(f"{e} {W.be_wr(BT[BT.era == e].y):.1f}%" for e in ERAS))

# parity of outcomes on MATCH vs the stored LOWVWAP walk
LW = pd.read_csv(W.LOWVWAP_ROWS, low_memory=False)
LW = LW[(LW.block == "BT") & (LW.seed == 1)].copy()
LW["T"] = (LW.te // W.BAR_MS) * W.BAR_MS - W.BAR_MS
J = B[B.MATCH].merge(LW[["pair", "T", "y", "why", "E"]], on=["pair", "T"], suffixes=("", "_lw"))
P(f"\nOutcome parity on MATCH vs WILLY_LOWVWAP rows (replay E, same walker): {len(J)} joined · same exit reason {np.mean(J.why == J.why_lw) * 100:.1f}% · "
  f"|Δy| ≤ 0.05 on {np.mean(abs(J.y - J.y_lw) <= 0.05) * 100:.1f}% · my E / replay E median {np.nanmedian(J.E / J.E_lw):.5f} · "
  f"avg y this study {J.y.mean():+.3f} vs LOWVWAP {J.y_lw.mean():+.3f}")

P("\n## Headline — today's WILLY on rebuilt signals (RH ticks)\n")
P("| set | era | N | WR | avg % | Σ %-pts | stop share | P(avg>0) day-boot |")
P("|---|---|---|---|---|---|---|---|")
for nm, D in (("ALL tick", BT), ("MATCH tick", BT[BT.MATCH]), ("R1 all (1m)", B[np.isfinite(B.y1)].assign(y=lambda d: d.y1, why=lambda d: d.why1))):
    for e in ERAS + ("all",):
        s = D if e == "all" else D[D.era == e]
        if not len(s):
            continue
        pp = W.boot_mean(s.y.values, s.day.values, n=2000)[0]
        P(f"| {nm} | {e} | {len(s)} | {(s.y > 0).mean() * 100:.1f}% | {f(s.y.mean())} | {s.y.sum():+.1f} | {(s.why == 'STOP').mean() * 100:.1f}% | {pp:.2f} |")
ms_m = MS[MS.block == "MASTER"]; ms_l = MS[MS.block == "LIVE"]
P(f"| MASTER 19 | as traded | {len(ms_m)} | {(ms_m.pct_traded > 0).mean() * 100:.0f}% | {f(ms_m.pct_traded.mean())} | {ms_m.pct_traded.sum():+.2f} (${ms_m.usd_traded.sum():+,.0f}) | | |")
P(f"| MASTER 19 | today re-priced | {len(ms_m)} | {(ms_m.y > 0).mean() * 100:.0f}% | {f(ms_m.y.mean())} | {ms_m.y.sum():+.2f} (${ms_m.usd_today.sum():+,.0f}) | | |")
P(f"| LIVE 8 | as traded | {len(ms_l)} | {(ms_l.pct_traded > 0).mean() * 100:.0f}% | {f(ms_l.pct_traded.mean())} | {ms_l.pct_traded.sum():+.2f} (${ms_l.usd_traded.sum():+,.0f}) | | |")
P(f"| LIVE 8 | today 1m re-price | {len(ms_l)} | {(ms_l.y > 0).mean() * 100:.0f}% | {f(ms_l.y.mean())} | {ms_l.y.sum():+.2f} (${ms_l.usd_today.sum():+,.0f}) | | |")

# ───────────── W0 / W1 paired alternatives
P("\n## W0 / W1 — does the selection add anything? (paired per episode, RH ticks both sides)\n")
Ae = A[A.T_ent.notna()].copy()
Ae["is_ent"] = Ae["T"] == Ae.T_ent
ent = Ae[Ae.is_ent & Ae.tick][["pair", "spike_ts", "y", "day", "era"]].rename(columns={"y": "y_w"})
At = Ae[Ae.tick & ~Ae.is_ent]
alts = {
    "TRIG (trigger bar close, no colour rule)": At[At.k == 0],
    "FIRST_GREEN (first green bar)": At[~At.red].sort_values("k").drop_duplicates(["pair", "spike_ts"]),
    "RED2 (2nd red bar)": At[At.red_ord == 2],
    "ANYRED (mean of the other red bars)": At[At.red],
    "RANDOM (mean of every other window bar)": At,
    "LATE_RANDOM (mean of bars after the WILLY bar)": At[At["T"] > At.T_ent],
}
P("| baseline | era | N episodes | WILLY avg | baseline avg | Δ / episode | P(Δ>0) day-boot | 95 % CI | WILLY WR | baseline WR |")
P("|---|---|---|---|---|---|---|---|---|---|")
W0 = []
for nm, X in alts.items():
    g = X.groupby(["pair", "spike_ts"]).agg(y_b=("y", "mean"), wr_b=("y", lambda s: (s > 0).mean())).reset_index()
    J = ent.merge(g, on=["pair", "spike_ts"])
    J["d"] = J.y_w - J.y_b
    for e in ERAS + ("all",):
        s = J if e == "all" else J[J.era == e]
        if len(s) < 5:
            continue
        pp, lo, hi = W.boot_mean(s.d.values, s.day.values, n=2000)
        P(f"| {nm} | {e} | {len(s)} | {f(s.y_w.mean())} | {f(s.y_b.mean())} | {f(s.d.mean())} | {pp:.2f} | [{f(lo)}, {f(hi)}] | "
          f"{(s.y_w > 0).mean() * 100:.0f}% | {s.wr_b.mean() * 100:.0f}% |")
        W0.append(dict(base=nm, era=e, n=len(s), d=s.d.mean(), p=pp))
# all armed episodes (incl. expired) — unpaired random-bar baseline
P("\nUnpaired — every window bar of every ARMED episode (incl. the 2,376 that never entered), RH ticks:\n")
P("| set | era | N bars | avg | WR | stop share |")
P("|---|---|---|---|---|---|")
for nm, X in (("all window bars", A[A.tick]), ("red bars", A[A.tick & A.red]), ("green bars", A[A.tick & ~A.red]),
              ("k = 0 trigger bars", A[A.tick & (A.k == 0)]), ("WILLY entries", A[A.tick & (A["T"] == A.T_ent)])):
    for e in ERAS:
        s = X[X.era == e]
        P(f"| {nm} | {e} | {len(s)} | {f(s.y.mean())} | {(s.y > 0).mean() * 100:.0f}% | {(s.why == 'STOP').mean() * 100:.0f}% |")
# drift / bracket benchmark
P("\nZero-drift benchmark: for a driftless price path, P(+1.25 before −4) = 4 / 5.25 = **76.2 %**; the sleeve breakeven WR (fees, slips, the late"
  f" TP, caps) is {BE:.1f} %. A WR in the high 70s is what a coin-flip path produces with this bracket.")

bucket_table(BT, "ent_bar_ret", [-99, -2.0, -0.75, 0], ["DEEP ≤ −2.0", "MID", "SHALLOW > −0.75"], "W1 — depth of the red bar")
bucket_table(BT, "wait_bars", [-1, 0, 1, 3, 99], ["0", "1", "2–3", "≥ 4"], "W4 — wait bars from arming to the red bar")

# ───────────── W2 / W3 / W4 / W5 / W6 / W6b / W7 buckets
bucket_table(BT, "red_vs_spike_q", [0, 0.5, 1.0, 999], ["< 0.5", "0.5–1", "≥ 1"], "W2 — red-bar volume ÷ spike-bar volume")
bucket_table(BT, "tbs_red", [0, 45, 50, 55, 100], ["< 45", "45–50", "50–55", "≥ 55"], "W2 — red-bar taker-buy share %")
bucket_table(BT, "tbs_red_last1m", [0, 40, 50, 60, 100], ["< 40", "40–50", "50–60", "≥ 60"], "W2 — last 1m of the red bar taker-buy share %")
bucket_table(BT, "tbs_spk", [0, 50, 55, 60, 100], ["< 50", "50–55", "55–60", "≥ 60"], "W2 — spike-bar taker-buy share %")
bucket_table(BT, "spike_conc", [0, 0.4, 0.6, 1.01], ["staircase ≤ 0.4", "mixed", "one-candle ≥ 0.6"], "W3 — spike shape")
bucket_table(BT, "spike_uwick", [-0.01, 0.15, 0.4, 1.01], ["< 0.15", "0.15–0.4", "≥ 0.4"], "W3 — spike-bar upper wick share")
bucket_table(BT, "spike_ret6", [0, 6.5, 10, 9999], ["5–6.5", "6.5–10", "≥ 10"], "W3 — spike size (30-min %)")
bucket_table(BT, "spike_volx", [0, 25, 40, 1e9], ["20–25×", "25–40×", "≥ 40×"], "W3 — spike volume multiple")
bucket_table(BT, "off_high", [-999, -5, -2.5, 0], ["≤ −5", "−5…−2.5", "> −2.5"], "W3 — entry vs the spike high %")
bucket_table(BT, "hour", [-1, 7, 12, 20, 23], ["Asia 0–8", "EU 8–13", "US 13–21", "late 21–24"], "W4 — session (UTC hour of the red bar)")
bucket_table(BT, "vs_vwap", [-999, 0, 999], ["≤ 0 (below VWAP)", "> 0"], "W5 — entry vs spike VWAP (sign)")
bucket_table(BT, "retrace_frac", [-1, 0.2, 0.35, 99], ["< 0.2", "0.2–0.35", "≥ 0.35"], "W5 — share of the pump retraced")
bucket_table(BT, "vs_base_pct", [-999, 8, 12, 99999], ["< 8", "8–12", "≥ 12"], "W5 — entry vs pre-spike base %")
for col, ttl in (("btc_1h_slope", "W6 — BTC 1h EMA20 slope (sign)"), ("btc_15m", "W6 — BTC 15-min return (sign)"),
                 ("btc_1h", "W6 — BTC 1-hour return (sign)"), ("btc_bar", "W6b — BTC SAME 5m bar (sign)"),
                 ("eth_bar", "W6b — ETH same bar (sign)"), ("sol_bar", "W6b — SOL same bar (sign)")):
    bucket_table(BT, col, [-999, 0, 999], ["≤ 0", "> 0"], ttl)
bucket_table(BT, "btc_bar", [-999, -0.02, 0.02, 999], ["(ii) BETA: BTC red < −0.02", "(iii) FLAT", "(i) IDIO: BTC green > +0.02"], "W6b — class of the red candle")
bucket_table(BT, "brd_red", [-1, 40, 60, 101], ["≤ 40 % red (market green)", "40–60", "≥ 60 % red (market red)"], "W6b — top-50 share red on the same bar")
bucket_table(BT, "brd_1h", [-1, 50, 101], ["< 50 % up 1h", "≥ 50"], "W6 — top-50 breadth over 1 h")
bucket_table(BT, "mkt_tbs", [0, 48, 52, 100], ["< 48", "48–52", "≥ 52"], "W6b — market (BTC+ETH+SOL) 1m taker-buy share in the bar")
bucket_table(BT, "R_ent", [0, 0.2, 0.5, 1.0], ["< 0.2", "0.2–0.5", "0.5–1"], "W7 — turnover vol/mcap (live < 1)")
bucket_table(BT, "mcap", [0, 100e6, 500e6, 1e13], ["< $100M", "$100–500M", "≥ $500M"], "W7 — market cap (supply proxy × price)")
bucket_table(BT, "t50", [-1, 0.5, 2], ["not top-50", "top-50"], "W7 — previous-day top-50")
bucket_table(BT, "listing_days", [0, 180, 365, 99999], ["< 180 d", "180–365", "≥ 365"], "W7 — listing age")

# W6b die rate per class (stop share) and post-entry market path by outcome
P("\n### W6b — die rate per class (stop share) vs breakeven, RH ticks\n")
P("| class | era | N | stop share | TP share | avg | WR vs BE |")
P("|---|---|---|---|---|---|---|")
cls = pd.cut(BT.btc_bar, [-999, -0.02, 0.02, 999], labels=["BETA", "FLAT", "IDIO"])
for c in ("BETA", "FLAT", "IDIO"):
    for e in ERAS + ("all",):
        s = BT[(cls == c) & ((BT.era == e) if e != "all" else True)]
        if len(s):
            P(f"| {c} | {e} | {len(s)} | {(s.why == 'STOP').mean() * 100:.1f}% | {s.why.isin(['TP', 'LATE_TP']).mean() * 100:.1f}% | {f(s.y.mean())} | "
              f"{(s.y > 0).mean() * 100:.1f}% vs {BE:.1f}% |")
P("\n### W6b POST-ENTRY (descriptive — after entry, never a filter): market path by outcome, RH ticks\n")
P("| outcome | N | BTC +5 min | BTC +15 | BTC +30 | top-50 +5 | +15 | +30 | coin +5 | coin +15 | coin +30 |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for w_ in ("TP", "LATE_TP", "STOP", "CAP"):
    s = BT[BT.why == w_]
    P(f"| {w_} | {len(s)} | {f(s.btc_f5.mean())} | {f(s.btc_f15.mean())} | {f(s.btc_f30.mean())} | {f(s.brd_f5.mean())} | {f(s.brd_f15.mean())} | "
      f"{f(s.brd_f30.mean())} | {f(s.r5.mean())} | {f(s.r15.mean())} | {f(s.r30.mean())} |")
P("\nStop rate by the BTC move in the 15 min AFTER entry (descriptive):\n")
P("| BTC next 15 min | N | stop share | avg | stops' share of all stops |")
P("|---|---|---|---|---|")
bq = pd.cut(BT.btc_f15, [-99, -0.3, -0.1, 0.1, 0.3, 99], labels=["≤ −0.3", "−0.3…−0.1", "flat ±0.1", "0.1…0.3", "≥ 0.3"])
nst = (BT.why == "STOP").sum()
for lab in bq.cat.categories:
    s = BT[bq == lab]
    P(f"| {lab} | {len(s)} | {(s.why == 'STOP').mean() * 100:.1f}% | {f(s.y.mean())} | {(s.why == 'STOP').sum() / max(nst, 1) * 100:.0f}% |")
st = BT[BT.why == "STOP"]
P(f"\nStopped trades: coin −{abs(st.r15.mean()):.2f} % at +15 min vs BTC {f(st.btc_f15.mean())} % and top-50 {f(st.brd_f15.mean())} % — "
  f"share of stops with BTC down > 0.3 % in the next 30 min: {(st.btc_f30 < -0.3).mean() * 100:.0f}% (all entries {(BT.btc_f30 < -0.3).mean() * 100:.0f}%).")

# ───────────── W7 the live turnover filter itself (R ≥ 1 refused cohort)
P("\n## W7 — the live turnover filter: what it refuses (first red bar with R ≥ 1, priced as an entry), RH ticks\n")
Ar = A[A.tick & (A["T"] == A["T"]) & A.is_red1 & (A.R >= 1.0)]
Ak = A[A.tick & (A["T"] == A.T_ent)]
P("| set | era | N | WR | avg | P(avg>0) |")
P("|---|---|---|---|---|---|")
for nm, X in (("refused R ≥ 1 (first red)", Ar), ("kept (WILLY entries)", Ak)):
    for e in ERAS + ("all",):
        s = X if e == "all" else X[X.era == e]
        if len(s) > 3:
            P(f"| {nm} | {e} | {len(s)} | {(s.y > 0).mean() * 100:.0f}% | {f(s.y.mean())} | {W.boot_mean(s.y.values, s.day.values, n=1000)[0]:.2f} |")

# ───────────── W8 exit fit
P("\n## W8 — exit fit (WILLY entries; RH path fully tick-covered for 240 min)\n")
G = B[(B.src == "ticks") & B.tick]
P(f"N {len(G)} · MFE90 median {G.mfe90.median():+.2f} · MAE90 median {G.mae90.median():+.2f} · reached +1.25 within 90 min {np.isfinite(G.t_tp).mean() * 100:.0f}% "
  f"(median {G.t_tp.median():.1f} min) · touched −4 within 90 {np.isfinite(G.t_sl).mean() * 100:.0f}%")
stp = G[G.why == "STOP"]
P(f"Stops: {len(stp)} · fast crash (≤ 10 min) {(stp.t_sl <= 10).mean() * 100:.0f}% · 10–30 min {((stp.t_sl > 10) & (stp.t_sl <= 30)).mean() * 100:.0f}% · "
  f"> 30 min {(stp.t_sl > 30).mean() * 100:.0f}% · stopped trades' 90-min close (no stop) median {stp.cl90.median():+.2f} / mean {stp.cl90.mean():+.2f} "
  f"(recovered above −4: {(stp.cl90 > -4).mean() * 100:.0f}%, back to ≥ 0: {(stp.cl90 >= 0).mean() * 100:.0f}%)")
P("\nP&L decomposition (today's exit): " + " · ".join(f"{w_} {(G.why == w_).mean() * 100:.0f}% of trades, Σ {G[G.why == w_].y.sum():+.0f}" for w_ in ("TP", "LATE_TP", "STOP", "CAP")))
gcols = [c for c in G.columns if c.startswith("g_")]
P("\nExit grid — avg % per trade by era (plain TP / SL / cap, no late TP; 'none' = no stop); WILLY entries vs the RANDOM window bar "
  "(mean of every other window bar of the same episodes, tick-covered):\n")
Ar_ = A[A.tick & (A.src == "ticks") & A.T_ent.notna() & (A["T"] != A.T_ent)]
Rg = Ar_.groupby(["pair", "spike_ts"])[gcols].mean()
Gk = G.set_index(["pair", "spike_ts"])
common = Gk.index.intersection(Rg.index)
gr = []
for c in gcols:
    _, tp, sl, cap = c.split("_")
    row = dict(tp=float(tp), sl=float(sl), cap=int(cap))
    for e in ERAS:
        s = G[G.era == e][c]
        row[f"W_{e}"] = s.mean()
    row["W_all"] = G[c].mean()
    ce = Gk.loc[common]
    row["W_paired"] = ce[c].mean(); row["R_paired"] = Rg.loc[common][c].mean()
    gr.append(row)
GR = pd.DataFrame(gr)
GR["min_era"] = GR[[f"W_{e}" for e in ERAS]].min(axis=1)
today = dict(W_all=G.y.mean(), **{f"W_{e}": G[G.era == e].y.mean() for e in ERAS})
P(f"TODAY (+1.25 / −4 / 90 + late TP): " + " · ".join(f"{k[2:]} {f(v)}" for k, v in today.items()))
P("\nTop 12 cells by the WORST era (robustness), and the 1.25 / 4 / 90 plain reference:\n")
P("| TP | SL | cap | 2025 | 2026-H1 | 2026-H2 | all | worst era | paired WILLY | paired RANDOM bar | WILLY − RANDOM |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
show = pd.concat([GR.sort_values("min_era", ascending=False).head(12), GR[(GR.tp == 1.25) & (GR.sl == 4) & (GR.cap == 90)]])
for _, r in show.iterrows():
    P(f"| {r.tp:g} | {'none' if r.sl == 0 else f'{r.sl:g}'} | {int(r.cap)} | {f(r['W_2025'])} | {f(r['W_2026-H1'])} | {f(r['W_2026-H2'])} | "
      f"{f(r.W_all)} | {f(r.min_era)} | {f(r.W_paired)} | {f(r.R_paired)} | {f(r.W_paired - r.R_paired)} |")
beat = GR[(GR[[f"W_{e}" for e in ERAS]].values > np.array([today[f"W_{e}"] for e in ERAS])).all(axis=1)]
P(f"\nCells beating TODAY in all three eras: {len(beat)} of {len(GR)}" + (": " + ", ".join(f"{r.tp:g}/{r.sl:g}/{r.cap}" for r in beat.itertuples()) if len(beat) else ""))
P(f"Share of the 140 cells with a positive average in ALL three eras: {(GR.min_era > 0).mean() * 100:.0f}% · WILLY − RANDOM averaged over all cells: "
  f"{(GR.W_paired - GR.R_paired).mean():+.3f}")
GR.to_csv(os.path.join(W.OUT, "w8_grid.csv"), index=False)
P("\n**Deployable cells only** (live per-order exchange backstop = stop + 0.5 ≤ 4.5 % at 10× and liquidation ≈ −10 % → SL 6 / none are NOT "
  "deployable at 10×). Paired Δ vs TODAY per trade (same entries), day-bootstrap P(Δ>0) per era; MATCH = bar-exact engine fills:\n")
P("| TP | SL | cap | Δ 2025 (P) | Δ 2026-H1 (P) | Δ 2026-H2 (P) | Δ all | Δ MATCH | avg all |")
P("|---|---|---|---|---|---|---|---|---|")
dep = GR[(GR.sl > 0) & (GR.sl <= 4)].copy()
dep["d_all"] = [(G[f"g_{r.tp:g}_{r.sl:g}_{int(r.cap)}"] - G.y).mean() for r in dep.itertuples()]
for _, r in dep.sort_values("d_all", ascending=False).head(10).iterrows():
    c = f"g_{r.tp:g}_{r.sl:g}_{int(r.cap)}"
    cells = []
    for e in ERAS:
        s = G[G.era == e]
        d = (s[c] - s.y).values
        cells.append(f"{f(d.mean())} ({W.boot_mean(d, s.day.values, n=1000)[0]:.2f})")
    m = G[G.MATCH]
    P(f"| {r.tp:g} | {r.sl:g} | {int(r.cap)} | " + " | ".join(cells) + f" | {f(r.d_all)} | {f((m[c] - m.y).mean())} | {f(G[c].mean())} |")


# ───────────── GATES
P("\n## BLOCK cells — locked expectancy filter bar + walk-forward (RH ticks pooled; Δ = refuse change per signal)\n")
GT = eval_all(BT, B)
GT.to_csv(os.path.join(W.OUT, "gates.csv"), index=False)
P("| cell | scored | blocked N | WR | avg | P(avg<0) | days | max day / pair loss share | Δ 2025 | Δ 2026-H1 | Δ 2026-H2 | Δ MATCH | Δ R1 | gates ①②③④ WF | PASS |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in GT.iterrows():
    if not r.N:
        P(f"| {r.cell} | {r.scored} | 0 | | | | | | | | | | | | |"); continue
    g = "".join("✓" if x else "✗" for x in (r.g1_wr, r.g2_conf, r.g3_conc, r.g4_n, r.wf))
    P(f"| {r.cell} | {r.scored} | {r.N} | {r.wr:.1f}% | {f(r.avg)} | {r.p_neg:.2f} | {r.days} | {f(r.max_day_share, 2, False)} / {f(r.max_pair_share, 2, False)} | "
      f"{f(r['d_2025'])} | {f(r['d_2026-H1'])} | {f(r['d_2026-H2'])} | {f(r.d_match)} | {f(r.d_r1)} | {g} | {'**PASS**' if r.PASS else 'no'} |")
P(f"\nBE WR used for gate ①: {BE:.1f} %. Δ in %-points per signal of the whole cohort (refusing the blocked set). Haircut 30–50 % applies to any PASS.")

# null: within-era shuffle of y (day labels kept)
rng = np.random.default_rng(11)
npass = []
for it in range(int(os.environ.get("NULL_N", "60"))):
    S2 = BT.copy()
    for e in ERAS:
        ix = S2.index[S2.era == e]
        S2.loc[ix, "y"] = rng.permutation(S2.loc[ix, "y"].values)
    S2["y1"] = S2.y
    g2 = eval_all(S2, S2.assign(y1=S2.y), nboot=300)
    npass.append(int(g2.get("PASS", pd.Series(dtype=bool)).fillna(False).sum()))
P(f"\nShuffle null ({len(npass)} within-era permutations of y; nboot 300): expected PASS cells by luck {np.mean(npass):.2f} (max {max(npass)}); "
  f"observed PASS {int(GT.PASS.fillna(False).sum())} of {len(GT)}.")

# ───────────── MASTER / LIVE per rule
P("\n## MASTER (19, per batch) and LIVE (8) — every BLOCK cell, before / after (as traded and today re-priced), % and $\n")
ML = pd.concat([ms_rule_table(n) for n in BLOCKS], ignore_index=True)
ML.to_csv(os.path.join(W.OUT, "ms_rules.csv"), index=False)
P("| cell | block | batch | n | scored | blocked | as traded %-pts → rule | as traded $ → rule | today %-pts → rule | today $ → rule | blocked fills (as traded %) |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for r in ML.itertuples():
    P(f"| {r.cell} | {r.block} | {r.batch} | {r.n} | {r.scored} | {r.blocked} | {r.pct_traded:+.2f} → {r.pct_traded_rule:+.2f} | "
      f"{r.usd_traded:+,.0f} → {r.usd_traded_rule:+,.0f} | {r.pct_today:+.2f} → {r.pct_today_rule:+.2f} | {r.usd_today:+,.0f} → {r.usd_today_rule:+,.0f} | {r.blocked_list} |")
P("\nPer-trade master / live feature sheet:\n")
cols = ["block", "batch", "pair", "opened_at", "pct_traded", "usd_traded", "y", "why", "ent_bar_ret", "wait_bars", "red_vs_spike_q", "tbs_red",
        "spike_conc", "spike_uwick", "off_high", "vs_vwap", "retrace_frac", "btc_1h_slope", "btc_bar", "brd_red", "mkt_tbs", "btc_f15", "R_ent"]
X_ = MS[[c for c in cols if c in MS]].round(3)
P("| " + " | ".join(X_.columns) + " |"); P("|" + "---|" * len(X_.columns))
for _, r_ in X_.iterrows():
    P("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r_.values) + " |")

open(os.path.join(W.OUT, "tables.md"), "w").write("\n".join(OUT))
print("\n".join(OUT[:60]))
print("… written", os.path.join(W.OUT, "tables.md"))
