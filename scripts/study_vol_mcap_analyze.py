#!/usr/bin/env python3
"""📊 Oct-8 research — does turnover R = vol24h / market cap at entry separate FRENZY (and FRENZY_WILLY) winners from losers?

Inputs (no network): <scratch>/vm/{frenzy_gated,frenzy_today,frenzy_today_bear,willyA,willyB}.csv (study_vol_mcap_cohorts.py +
study_vol_mcap_willy_price.py), reports/MASTER_POOL_stacked.csv + the live export (real fills), reports/cache_vol_mcap/ (CoinGecko
daily market_caps / total_volumes; pass-1 mapping.json, accepted per pair only when CG price × mult matches the pair 5m close), reports/backtest_cache/k5m_full (+ scratch flag_x/k5new for Oct 4–8).
R variants: R (PRIMARY, the bot's own source) = Binance futures 24h quote volume (288 closed 5m bars before entry; = the stamped
entry_pair_volume_24h_usd on real fills) / mcap_b, mcap_b = the Binance Info-panel (CoinMarketCap) market cap fetched today
(scripts/study_vol_mcap_bapi.py) × (pair close at entry / pair price today) — i.e. TODAY's circulating supply × entry price
(ASSUMPTION: supply ≈ constant; checked vs stamped fills and vs CoinGecko history) · on real fills R_live = the stamps ·
secondary: R_fut = Binance FUTURES quote volume of the 288 closed 5m bars before entry / CG mcap (last daily point ≤ entry, ≤ 2 days
old) · R_cg = CG all-exchange total_volume at that same point / CG mcap · R_live (real fills only) = stamped
entry_pair_volume_24h_usd / stamped entry_mcap_usd.
Writes reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv and prints the markdown tables (the report embeds them).
Usage: S=<scratch> venv/bin/python scripts/study_vol_mcap_analyze.py > <scratch>/vm/tables.md"""
import json, os, sys
import numpy as np, pandas as pd

S = os.environ["S"]; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
VM = os.path.join(S, "vm"); CD = "reports/cache_vol_mcap"; K5 = "reports/backtest_cache/k5m_full"; K5N = os.path.join(S, "flag_x/k5new")
LIVE = "/Users/guillearslanian/Downloads/scalpars_orders_paper_2026-10-08_12-57-28.csv"
RNG = np.random.default_rng(20261008); NB = 4000; NNULL = 1000
MAP = json.load(open(os.environ.get("MAPF", f"{CD}/mapping.json")))
BAPI = json.load(open(f"{CD}/bapi_supply.json")); PNOW = json.load(open(f"{CD}/binance_ticker_price.json"))

# ── data assembly ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_ch, _k5, _CGV = {}, {}, {}


def cg_series(pair):
    m = MAP.get(pair) or {}
    if not m.get("id"):
        return None
    if pair not in _ch:
        f = f"{CD}/chart/{m['id']}.json"
        if not os.path.exists(f):
            _ch[pair] = None
        else:
            j = json.load(open(f))
            a = pd.DataFrame(j["market_caps"], columns=["t", "mc"]).merge(pd.DataFrame(j["total_volumes"], columns=["t", "tv"]), on="t") \
                .merge(pd.DataFrame(j["prices"], columns=["t", "p"]), on="t")
            a["t"] = a.t.astype("int64"); a = a.sort_values("t").reset_index(drop=True)
            k = k5(pair); ok = False          # historical price verification: CG price × mult vs the pair's own 5m close
            if k is not None:
                i = np.searchsorted(k.open_time.values, a.t.values, side="right") - 1
                g = (i >= 0) & (a.t.values - k.open_time.values[np.maximum(i, 0)] <= 600_000)
                r = a.p.values[g] * m.get("mult", 1) / k.c.values[i[g]]
                ok = len(r) >= 20 and 0.8 <= np.median(r) <= 1.25 and ((r > 1 / 1.5) & (r < 1.5)).mean() >= 0.8
            _CGV[pair] = ok
            _ch[pair] = a if ok else None
    return _ch[pair]


def k5(pair):
    if pair not in _k5:
        parts = []
        f = f"{K5}/{pair}.csv"
        if os.path.exists(f):
            parts.append(pd.read_csv(f, usecols=["open_time", "c", "qvol"]))
        g = f"{K5N}/{pair}.npy"
        if os.path.exists(g):
            a = np.load(g); parts.append(pd.DataFrame(dict(open_time=a[:, 0].astype("int64"), c=a[:, 4], qvol=a[:, 6])))
        _k5[pair] = (pd.concat(parts).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True) if parts else None)
    return _k5[pair]


def features(df, tcol="t0"):
    mc, tv, vf, age, mb = [], [], [], [], []
    for p, t in zip(df.pair, df[tcol].astype("int64")):
        c = cg_series(p)
        if c is None:
            mc.append(np.nan); tv.append(np.nan); age.append(np.nan)
        else:
            i = np.searchsorted(c.t.values, t, side="right") - 1
            if i < 0 or t - c.t.values[i] > 2 * 86_400_000 or not (c.mc.values[i] > 0):
                mc.append(np.nan); tv.append(np.nan); age.append(np.nan)
            else:
                mc.append(c.mc.values[i]); tv.append(c.tv.values[i]); age.append((t - c.t.values[i]) / 3.6e6)
        k = k5(p); b = BAPI.get(p) or {}
        if k is None:
            vf.append(np.nan); mb.append(np.nan); continue
        j = np.searchsorted(k.open_time.values, t - 300_000, side="right")   # bars with open ≤ t − 5 min are closed at t
        px_t = float(k.c.values[j - 1]) if j >= 1 and k.open_time.values[j - 1] >= t - 600_000 else np.nan
        mb.append(b["mc"] * px_t / PNOW[p] if b.get("mc") and PNOW.get(p) and np.isfinite(px_t) else np.nan)
        if j < 288 or k.open_time.values[j - 1] < t - 600_000 or k.open_time.values[j - 288] < t - 300_000 - 288 * 300_000 - 3_600_000:
            vf.append(np.nan); continue
        vf.append(float(k.qvol.values[j - 288:j].sum()))
    df = df.copy(); df["mcap_b"] = mb; df["mcap_cg"] = mc; df["vol_cg"] = tv; df["vol_fut"] = vf; df["mcap_age_h"] = age
    df["R_fut"] = df.vol_fut / df.mcap_cg; df["R_cg"] = df.vol_cg / df.mcap_cg
    df["lR_fut"] = np.log10(df.R_fut); df["lR_cg"] = np.log10(df.R_cg)
    df["R"] = df.vol_fut / df.mcap_b; df["lR"] = np.log10(df.R)
    return df


def real_fills():
    from build_master_pool import frenzy_fixed_pct, frenzy_bearish_stack_block
    P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    L = pd.read_csv(LIVE, low_memory=False)
    fz = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
    P = P[P.entry_strategy.isin(fz) & (P.status == "CLOSED") & (P.stack_block_reason.astype(str) == "FRENZY_SLEEVE")].assign(src="master")
    L = L[L.entry_strategy.isin(fz) & (L.status == "CLOSED")].assign(src="live")
    A = pd.concat([P, L], ignore_index=True)
    A["k"] = A.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + A.pair + "|" + A.direction
    A = A.drop_duplicates("k", keep="first")
    A["pct"] = [frenzy_fixed_pct(s, pk, pc) for s, pk, pc in zip(A.entry_strategy, pd.to_numeric(A.peak_pnl, errors="coerce"),
                                                                    pd.to_numeric(A.pnl_percentage, errors="coerce"))]
    A["bear"] = [frenzy_bearish_stack_block(s, r, g) for s, r, g in zip(A.entry_strategy, A.entry_btc_1d_ret_pct, A.entry_btc_trend_gap_pct)]
    A["t0"] = pd.to_datetime(A.opened_at.astype(str).str[:19]).astype("int64") // 10**6
    A["day"] = A.opened_at.astype(str).str[:10]; A["sleeve"] = A.entry_strategy
    A["R_live"] = pd.to_numeric(A.entry_pair_volume_24h_usd, errors="coerce") / pd.to_numeric(A.entry_mcap_usd, errors="coerce")
    A["lR_live"] = np.log10(A.R_live)
    A = features(A)
    A["mcap_b_entry"] = [BAPI.get(p, {}).get("mc", np.nan) * float(e) / PNOW[p] if PNOW.get(p) and BAPI.get(p, {}).get("mc") else np.nan
                         for p, e in zip(A.pair, pd.to_numeric(A.entry_price, errors="coerce"))]
    A["R_b"] = pd.to_numeric(A.entry_pair_volume_24h_usd, errors="coerce") / A.mcap_b_entry; A["lR_b"] = np.log10(A.R_b)
    return A


def supply_check():
    """stamped entry_mcap_usd (CMC via the bot's endpoint, at entry) vs today's endpoint mc × entry price / today's price, on EVERY
    stamped fill (any strategy) of the master pool + the live export — tests the constant-supply reconstruction where both exist."""
    P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False); L = pd.read_csv(LIVE, low_memory=False)
    A = pd.concat([P.assign(src="master"), L.assign(src="live")], ignore_index=True)
    A["k"] = A.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + A.pair + "|" + A.direction
    A = A.drop_duplicates("k"); A = A[pd.to_numeric(A.entry_mcap_usd, errors="coerce") > 0].copy()
    A["rec"] = [BAPI.get(p, {}).get("mc", np.nan) * float(e) / PNOW[p] if PNOW.get(p) and BAPI.get(p, {}).get("mc") else np.nan
                for p, e in zip(A.pair, pd.to_numeric(A.entry_price, errors="coerce"))]
    A["ratio"] = A.rec / pd.to_numeric(A.entry_mcap_usd, errors="coerce"); s = A[A.ratio.notna()]
    lr = np.log(s.ratio)
    return (f"Constant-supply check on {len(s)}/{len(A)} stamped fills ({s.pair.nunique()} pairs, opened {s.opened_at.min()[:10]} → "
            f"{s.opened_at.max()[:10]}): reconstructed / stamped mcap median {s.ratio.median():.3f}, IQR {s.ratio.quantile(.25):.3f}–"
            f"{s.ratio.quantile(.75):.3f}, within ±10 % {((s.ratio > 0.9) & (s.ratio < 1.1)).mean() * 100:.0f}%, within ±25 % "
            f"{((s.ratio > 0.8) & (s.ratio < 1.25)).mean() * 100:.0f}%, Spearman(rec, stamp) {sp(s.rec, s.entry_mcap_usd):+.3f}. "
            f"Worst: " + ", ".join(f"{p} {r:.2f}" for p, r in s.assign(a=lr.abs()).sort_values("a", ascending=False)[["pair", "ratio"]].head(5).values) + "\n")


def drift_check(T):
    """year-long supply drift: mcap_b (today's supply × entry price) / CoinGecko daily mcap at entry, by month."""
    s = T[(T.mcap_b > 0) & (T.mcap_cg > 0)].copy(); s["r"] = s.mcap_b / s.mcap_cg; s["m"] = s.day.str[:7]
    rows = ["| month | N | median mcap_b / mcap_CG | IQR | share within ±25 % |", "|---|---|---|---|---|"]
    for m, g in s.groupby("m"):
        rows.append(f"| {m} | {len(g)} | {g.r.median():.3f} | {g.r.quantile(.25):.2f}–{g.r.quantile(.75):.2f} | {((g.r > .8) & (g.r < 1.25)).mean() * 100:.0f}% |")
    return "\n".join(rows) + f"\n\nSpearman(log R, log R_fut[CG mcap]) = {sp(s.lR, s.lR_fut):+.3f} on {len(s)} fills.\n"


def sp(a, b):
    a = pd.Series(np.asarray(a, float)); b = pd.Series(np.asarray(b, float)); m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank()) if m.sum() > 2 else np.nan


def md(df):
    f = lambda v: (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


# ── statistics ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cboot(v, cl, nb=NB):
    v = np.asarray(v, float)
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    c = pd.factorize(np.asarray(cl))[0]; n = c.max() + 1; s = np.bincount(c, v, n); k = np.bincount(c, minlength=n)
    K = RNG.integers(0, n, (nb, n)); m = s[K].sum(1) / np.maximum(k[K].sum(1), 1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m < 0).mean()


def be_wr(v):
    v = np.asarray(v); w = v[v > 0]; l = v[v <= 0]
    return np.nan if not len(w) or not len(l) else abs(l.mean()) / (w.mean() + abs(l.mean())) * 100


def conc(T, key):
    s = T.groupby(key).pct.sum(); neg = -s[s < 0]
    if neg.sum() <= 0:
        return 0.0, ""
    top = neg.sort_values(ascending=False)
    return float(top.iloc[0] / neg.sum() * 100), str(top.index[0])


def row(T, label):
    v = T.pct.values
    if not len(v):
        return f"| {label} | 0 | | | | | | | | |"
    lo, hi, _ = cboot(v, T.day.values)
    c2 = -T.groupby("pair").pct.sum().sort_values().head(2).clip(upper=0).sum() if (v < 0).any() else 0
    negsum = -T.groupby("pair").pct.sum().clip(upper=0).sum()
    top2 = (c2 / negsum * 100) if negsum > 0 else 0
    return (f"| {label} | {len(v)} | {T.day.nunique()} | {T.pair.nunique()} | {(v > 0).mean() * 100:.0f}% | **{v.mean():+.3f}** | "
            f"[{lo:+.2f}, {hi:+.2f}] | {v.sum():+.1f} | {v.min():+.2f} | {top2:.0f}% |")


HDR = "| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |\n|---|---|---|---|---|---|---|---|---|---|"


def quant_table(T, col, q=5):
    T = T[T[col].notna()].copy()
    if len(T) < 2 * q:
        return "(too few scored fills)\n", None
    T["b"] = pd.qcut(T[col].rank(method="first"), q, labels=False)
    out = [HDR]; means = []
    for b, g in T.groupby("b"):
        rng = f"Q{b + 1} {10 ** g[col].min():.3g}–{10 ** g[col].max():.3g}" if col.startswith("lR") else f"Q{b + 1} {g[col].min():.3g}–{g[col].max():.3g}"
        out.append(row(g, rng)); means.append(g.pct.mean())
    rho = sp(means, range(len(means)))
    return "\n".join(out) + f"\n\nbucket-order Spearman (bucket index vs avg) = {rho:+.2f} · trade-level Spearman(R, P&L) = " \
        f"{sp(T[col], T.pct):+.3f}\n", means


def cuts_for(x, qs=np.arange(0.10, 0.91, 0.05)):
    return np.unique(np.quantile(x, qs))


def scan(x, y, cuts, minn):
    best = (0.0, None, None)
    for c in cuts:
        lo = y[x <= c]; hi = y[x > c]
        if len(lo) < minn or len(hi) < minn:
            continue
        d = hi.mean() - lo.mean()
        if abs(d) > abs(best[0]):
            best = (d, c, "low" if d > 0 else "high")    # side to BLOCK = the worse side
    return best


def null_p(x, y, cuts, minn, obs):
    if obs == 0:
        return np.nan
    cnt = 0
    for _ in range(NNULL):
        d, _, _ = scan(x, RNG.permutation(y), cuts, minn)
        cnt += abs(d) >= abs(obs)
    return (cnt + 1) / (NNULL + 1)


def side(T, col, c, blk):
    return T[T[col] <= c] if blk == "low" else T[T[col] > c]


def other(T, col, c, blk):
    return T[T[col] > c] if blk == "low" else T[T[col] <= c]


def expectancy_bar(T, col, c, blk):
    B = side(T, col, c, blk); K = other(T, col, c, blk)
    be = be_wr(K.pct.values); wr = (B.pct > 0).mean() * 100 if len(B) else np.nan
    lo, hi, pneg = cboot(B.pct.values, B.day.values)
    dsh, dname = conc(B, "day"); psh, pname = conc(B, "pair")
    checks = {"WR < kept breakeven WR": wr < be, "P(avg<0) ≥ 95% (day-clustered)": pneg >= 0.95, "≥ 8 distinct days": B.day.nunique() >= 8,
              "no day ≥ 50% of loss": dsh < 50, "no pair ≥ 50% of loss": psh < 50, "N ≥ 15": len(B) >= 15}
    txt = (f"blocked side = R {'≤' if blk == 'low' else '>'} {10 ** c:.3g} → N {len(B)}, WR {wr:.0f}% vs kept breakeven WR {be:.0f}%, "
           f"avg {B.pct.mean():+.3f} [{lo:+.2f}, {hi:+.2f}], P(avg<0) {pneg:.3f}, days {B.day.nunique()}, top day {dsh:.0f}% ({dname}), "
           f"top pair {psh:.0f}% ({pname}); kept N {len(K)} avg {K.pct.mean():+.3f}\n\n"
           + " · ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items())
           + f"\n\n**Bar {'PASSED' if all(checks.values()) else 'NOT passed'}.**")
    dsum = -B.pct.sum() / max(len(T), 1)
    txt += (f" In-sample Δ from blocking = {dsum:+.3f} %/fill of the whole cohort → after 30–50 % haircut {dsum * 0.5:+.3f}…{dsum * 0.7:+.3f}.")
    return txt, all(checks.values())


def oos(T, col, split, minn):
    out = []
    for a, b, lab in ((T[T.day < split], T[T.day >= split], f"pick < {split} → test ≥ {split}"),
                      (T[T.day >= split], T[T.day < split], f"pick ≥ {split} → test < {split}")):
        a = a[a[col].notna()]; b = b[b[col].notna()]
        if len(a) < 2 * minn or len(b) < 10:
            out.append(f"| {lab} | too few | | | |"); continue
        d, c, blk = scan(a[col].values, a.pct.values, cuts_for(a[col].values), minn)
        if c is None:
            out.append(f"| {lab} | no cut | | | |"); continue
        tb = side(b, col, c, blk); tk = other(b, col, c, blk)
        out.append(f"| {lab} | block {blk} R {'≤' if blk == 'low' else '>'} {10 ** c:.3g} (in-sample Δ {d:+.3f}) | {len(tb)} · "
                   f"{(tb.pct > 0).mean() * 100 if len(tb) else float('nan'):.0f}% · {tb.pct.mean() if len(tb) else float('nan'):+.3f} | "
                   f"{len(tk)} · {tk.pct.mean() if len(tk) else float('nan'):+.3f} | "
                   f"{'holds' if len(tb) and len(tk) and tb.pct.mean() < tk.pct.mean() else 'FAILS'} |")
    # leave-one-month-out
    T = T[T[col].notna()].copy(); T["m"] = T.day.str[:7]; hb, hk = [], []
    for m in sorted(T.m.unique()):
        a = T[T.m != m]; b = T[T.m == m]
        d, c, blk = scan(a[col].values, a.pct.values, cuts_for(a[col].values), minn)
        if c is None:
            continue
        hb.append(side(b, col, c, blk)); hk.append(other(b, col, c, blk))
    HB = pd.concat(hb) if hb else T.iloc[:0]; HK = pd.concat(hk) if hk else T.iloc[:0]
    lo, hi, _ = cboot(HB.pct.values, HB.day.values) if len(HB) > 1 else (np.nan, np.nan, np.nan)
    out.append(f"| leave-one-month-out ({T.m.nunique()} months) | per-month re-picked cut | {len(HB)} · {(HB.pct > 0).mean() * 100 if len(HB) else float('nan'):.0f}% · "
               f"{HB.pct.mean() if len(HB) else float('nan'):+.3f} [{lo:+.2f}, {hi:+.2f}] | {len(HK)} · {HK.pct.mean() if len(HK) else float('nan'):+.3f} | "
               f"{'holds' if len(HB) and HB.pct.mean() < HK.pct.mean() else 'FAILS'} |")
    return "| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |\n|---|---|---|---|---|\n" + "\n".join(out) + "\n"


def month_table(T, col):
    T = T.copy(); T["m"] = T.day.str[:7]; med = T[col].median(); rows = ["| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |", "|---|---|---|---|---|---|---|"]
    for m, g in T.groupby("m"):
        s = g[g[col].notna()]; lo = s[s[col] <= med].pct; hi = s[s[col] > med].pct
        rows.append(f"| {m} | {len(g)} | {len(s)} | {(g.pct > 0).mean() * 100:.0f}% | {g.pct.mean():+.3f} | {len(lo)} · {lo.mean() if len(lo) else float('nan'):+.3f} | "
                    f"{len(hi)} · {hi.mean() if len(hi) else float('nan'):+.3f} |")
    return "\n".join(rows) + f"\n\n(pooled median R = {10 ** med:.3g})\n"


def overlap(T, col, cols):
    out = ["| variable | Spearman with log R | N |", "|---|---|---|"]
    for c in cols:
        if c in T and pd.to_numeric(T[c], errors="coerce").notna().sum() > 10:
            s = T[[col, c]].apply(pd.to_numeric, errors="coerce").dropna()
            out.append(f"| {c} | {sp(s[col], s[c]):+.2f} | {len(s)} |")
    return "\n".join(out) + "\n"


def study(name, T, split, minn, ovl, be_note=""):
    print(f"\n## {name}\n")
    n0 = len(T)
    LAB = {"lR": "R — PRIMARY: Binance futures 24h quote vol / bot-source mcap (today's CMC supply × entry price)",
           "lR_fut": "R_fut — robustness: Binance futures 24h vol / CoinGecko daily mcap (history, no supply assumption)",
           "lR_cg": "R_cg — robustness: CoinGecko all-exchange 24h vol / CoinGecko mcap"}
    print("**Supply-drift check (bot-source mcap vs CoinGecko history)**\n"); print(drift_check(T))
    for col in ("lR", "lR_fut", "lR_cg"):
        if col not in T or T[col].notna().sum() == 0:
            continue
        sc = T[col].notna()
        print(f"\n### {name} — {LAB[col]}\n")
        print(f"Coverage: {sc.sum()}/{n0} fills scored ({sc.mean() * 100:.0f}%). Unscored fills: N {(~sc).sum()}, avg "
              f"{T[~sc].pct.mean() if (~sc).any() else float('nan'):+.3f} (NOT treated as 'rest'). Scored cohort: {row(T[sc], 'all scored')[2:]}\n")
        U = T[sc]
        if len(U) < 30:
            print("too few scored fills for buckets\n"); continue
        print(month_table(U, col))
        print("**Quintiles**\n"); qt, _ = quant_table(U, col, 5); print(qt)
        if len(U) >= 200:
            print("**Deciles**\n"); qt, _ = quant_table(U, col, 10); print(qt)
        x, y = U[col].values, U.pct.values; cuts = cuts_for(x)
        d, c, blk = scan(x, y, cuts, minn)
        if c is None:
            print("no admissible cut\n"); continue
        p = null_p(x, y, cuts, minn, d)
        print(f"**Threshold scan** ({len(cuts)} cuts at the 10–90 % quantiles, each side ≥ {minn}): best = block {blk.upper()} turnover, "
              f"R {'≤' if blk == 'low' else '>'} {10 ** c:.3g} (i.e. 24h volume = {10 ** c * 100:.1f} % of mcap); Δ(high − low) = {d:+.3f} %/fill. "
              f"Shuffled-label null ({NNULL}×, same scan, max |Δ|): **p = {p:.3f}**.\n")
        txt, ok = expectancy_bar(U, col, c, blk); print("**Expectancy bar on the blocked side**: " + txt + "\n")
        print("**Out-of-sample**\n"); print(oos(U, col, split, minn))
        if ovl:
            print("**Overlap with stamped / filtered variables**\n"); print(overlap(U, col, ovl))


def main():
    out_rows = []
    # real fills
    R = real_fills()
    print("## REAL FILLS (master FRENZY_SLEEVE kept + live export, deduped (opened_at, pair, direction), CLOSED, fixed +3/−3 pricing)\n")
    print(supply_check())
    cols = ["src", "opened_at", "pair", "sleeve", "pct", "bear", "entry_atr_pct", "entry_frenzy_gvol", "entry_pair_volume_24h_usd", "entry_mcap_usd", "R_live", "R_b", "R_cg"]
    print(md(R[cols].sort_values("opened_at")))
    for lab, M in (("all", R), ("bearish-day block applied", R[~R.bear])):
        for col in ("lR_live", "lR_b", "lR_cg"):
            s = M[M[col].notna()]
            if len(s) < 4:
                continue
            med = s[col].median(); lo = s[s[col] <= med]; hi = s[s[col] > med]
            print(f"\n{lab} · {col}: scored {len(s)}/{len(M)} · R ≤ median {10 ** med:.3g}: N {len(lo)} WR {(lo.pct > 0).mean() * 100:.0f}% avg {lo.pct.mean():+.3f} · "
                  f"R > median: N {len(hi)} WR {(hi.pct > 0).mean() * 100:.0f}% avg {hi.pct.mean():+.3f} · Spearman {sp(s[col], s.pct):+.2f}")
    out_rows.append(R[cols + ["day", "lR_live", "lR_b", "lR_cg", "vol_fut", "vol_cg", "mcap_b_entry", "mcap_cg"]].assign(cohort="REAL"))
    print("\nLive-stamp vs rebuilt R on the same real fills: "
          + ", ".join(f"{a}~{b} Spearman {sp(R[a], R[b]):+.2f} (N {R[[a, b]].dropna().shape[0]})" for a, b in (("R_live", "R_b"), ("R_live", "R_cg"))))

    OV = ["atr", "U2", "gvol", "vol24", "gain_pct", "run_pct", "hours", "bar_ret", "vol_mult", "above_streak", "btc_1d_ret", "btc_gap", "lmcap", "lvol"]
    for name, f, split, minn in (("FRENZY backtest — TODAY's kept stack WITHOUT the bearish block (ATR ≤ 3.0, sequenced, FIX +3/−3 @ 8 s)", "frenzy_today", "2026-06-01", 25),
                                 ("FRENZY backtest — TODAY's kept stack WITH the bearish-day block (live today)", "frenzy_today_bear", "2026-06-01", 25),
                                 ("FRENZY backtest — every gated signal, any ATR, unsequenced (larger N, includes ATR-refused)", "frenzy_gated", "2026-06-01", 25)):
        T = pd.read_csv(f"{VM}/{f}.csv").rename(columns={"t_signal_close": "t0", "FIX38": "pct"})
        T = features(T); T["lmcap"] = np.log10(T.mcap_b); T["lvol"] = np.log10(T.vol_fut)
        study(name, T, split, minn, OV)
        out_rows.append(T.assign(cohort=f))
    for name, f in (("FRENZY_WILLY entry A — first flag (+1 net / −3 / 60 min, real ticks)", "willyA"),
                    ("FRENZY_WILLY entry B — fresh ON bars not taken by today's FRENZY / WIDE", "willyB")):
        T = pd.read_csv(f"{VM}/{f}.csv"); T = T[T.st == "ok"].copy()
        T = features(T); T["lmcap"] = np.log10(T.mcap_b); T["lvol"] = np.log10(T.vol_fut)
        if "vol24" not in T:
            T["vol24"] = T.vol_fut
        study(name, T, "2026-06-01", 40, OV)
        out_rows.append(T.assign(cohort=f))
    O = pd.concat(out_rows, ignore_index=True)
    keep = [c for c in ["cohort", "src", "opened_at", "t0", "day", "pair", "sleeve", "pct", "bear", "atr", "U2", "vol24", "vol_fut", "vol_cg", "mcap_cg",
                        "entry_mcap_usd", "entry_pair_volume_24h_usd", "mcap_b", "mcap_b_entry", "R_live", "R_b", "R", "R_fut", "R_cg", "mcap_age_h", "why"] if c in O]
    O[keep].to_csv("reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv", index=False)


if __name__ == "__main__":
    main()
