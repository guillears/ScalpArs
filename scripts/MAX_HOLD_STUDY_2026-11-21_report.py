#!/usr/bin/env python3
"""MAX_HOLD_STUDY_2026-11-21 — statistics, tables and chart (research only).

Reads reports/MAX_HOLD_STUDY_2026-11-21_data/{bt_EXH, bt_DIST, bt_FB, willy}.pkl (built by _bt.py / _willy.py) + the DL 616 real-fill
re-walks (FUNDING_STUDY_ALL_2026-11-21_data/real_walk.pkl, real_cohort_final.csv) and the HT kept list.
Per sleeve × block × cap: N · WR · avg % net of funding · Σ % · $ at one size · DCR (calendar days, one size) · avg / median hold ·
funding avg % · Δ vs today (day-block bootstrap 95 % CI, P(better)) · Δ price-only (funding effect) · Δ at time-exit slip 0.02 / 0.30 ·
change split (SAVED / DEEPER / DRIFT for longer caps; CUT_WIN / SAVED_LOSS / EARLY_TIME for shorter) · LOO by pair and by day ·
top-pair / top-day share of Σ Δ · per half-year Δ. Untouched fills: max |Δ| reported (must be 0).
→ reports/MAX_HOLD_STUDY_2026-11-21_tables.md, _chart.png, _data/summary.csv
Usage: venv/bin/python -I scripts/MAX_HOLD_STUDY_2026-11-21_report.py"""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP = os.path.join(ROOT, "reports")
D = os.path.join(RP, "MAX_HOLD_STUDY_2026-11-21_data")
FD = os.path.join(RP, "FUNDING_STUDY_ALL_2026-11-21_data")
HR, DAY = 3_600_000, 86_400_000
SPLIT = pd.Timestamp("2025-07-01").value // 10**6
_sp = importlib.util.spec_from_file_location("FSC", os.path.join(ROOT, "scripts", "FUNDING_STOP_2026-11-21_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)

SLEEVES = {  # name: (caps, today, one-size X of equity, cap unit)
    "EXH": ((6, 8, 12, 16, 20, 24, 30, 36, 48), 24, 0.61, "h"),
    "DIST": ((6, 8, 12, 16, 20, 24, 30, 36, 48), 24, 0.61, "h"),
    "FB": ((24, 36, 48, 60, 72, 96, 120), 72, 0.61, "h"),
    "HT": ((4, 6, 8, 12, 16, 24), 24, 0.305, "h"),
    "WILLY": ((45, 60, 90, 120, 180), 90, 0.305, "min"),
}
EQ = 10_000.0


def half(ms):
    t = pd.Timestamp(int(ms), unit="ms")
    return f"{t.year}-H{1 if t.month <= 6 else 2}"


WHY = {"FRENZY_TP": "TP", "STOP_LOSS": "STOP", "MAX_HOLD_TIME": "TIME", "DATA_END": "OPEN", "WILLY_RUNNER_TRAIL": "RUNNER",
       "WILLY_LATE_TP": "LATE_TP"}


def norm(df):
    for c in [c for c in df.columns if c.startswith("c") and c.endswith("_why")]:
        df[c] = df[c].replace(WHY)
    return df


def load_bt(s):
    if s in ("EXH", "DIST", "FB"):
        X = pd.read_pickle(os.path.join(D, f"bt_{s}.pkl"))
        X = X[~X.miss_ext].copy()
        X["period"] = np.where(X.te < SPLIT, "DESIGN", "HOLDOUT")
        return X
    W = pd.read_pickle(os.path.join(D, "willy.pkl"))
    W = W[(W.sleeve == ("HT" if s == "HT" else "WILLY")) & (W.kind == "BT")].copy()
    return W


def load_real(s):
    """→ {block name: frame with c{H}_* columns}; real fills whose exit is before the shortest cap carry today's result on every cap."""
    caps, H0, _, _ = SLEEVES[s]
    out = {}
    if s in ("HT", "WILLY"):
        W = pd.read_pickle(os.path.join(D, "willy.pkl"))
        if s == "HT":
            K = pd.read_csv(os.path.join(RP, "KEPT_LIST_WILLY_HT_2026-11-21.csv"))
            R = W[(W.sleeve == "HT") & (W.kind == "REAL")].merge(K, on=["pair", "k"])
            out["MASTER (today's filters)"] = R[R.block == "MASTER"].copy()
            out["BATCH (today's filters)"] = R[R.block == "BATCH"].copy()
        else:
            R = W[(W.sleeve == "WILLY") & (W.kind == "REAL") & W.today_keep].copy()
            out["MASTER (stack_keep)"] = R
        return {k: norm(v) for k, v in out.items()}

    def const(df, net, why, hold, fund, te):
        f = pd.DataFrame(dict(pair=df.pair.values, te=te, label="", period="REAL"))
        for H in caps:
            f[f"c{H}_net"] = net; f[f"c{H}_why"] = why; f[f"c{H}_hold"] = hold; f[f"c{H}_fund"] = fund
            f[f"c{H}_n02"] = net; f[f"c{H}_n30"] = net
        return f

    RW = pd.read_pickle(os.path.join(FD, "real_walk.pkl"))
    if s == "EXH":
        e = RW[RW.sleeve == "EXHAUSTION_SHORT"]
        e = e[e.id != "ENSO(open)"]
        te = pd.to_datetime(e.opened_at).values.astype("datetime64[ms]").astype("int64")
        assert e.base_hold.max() < min(caps), "an EXH real fill reaches the shortest cap — walk it"
        out["BATCH (8 closed, today's exit)"] = const(e, e.base_net.values, e.base_why.values, e.base_hold.values, e.base_fund.values, te)
    elif s == "FB":
        e = RW[RW.sleeve == "FLATBASE_SHORT"]
        assert e.base_hold.max() < min(caps)
        for b, nm in (("MASTER", "MASTER (17 kept)"), ("BATCH", "BATCH (3 kept)")):
            x = e[e.block == b]
            out[nm] = const(x, x.base_net.values, x.base_why.values, x.base_hold.values, x.base_fund.values, x.te.values)
    elif s == "DIST":
        r = pd.read_csv(os.path.join(FD, "real_cohort_final.csv"), low_memory=False)
        x = r[r.entry_strategy == "DISTRIBUTION_SHORT"]
        assert x.hold_h.max() < min(caps)
        te = pd.to_datetime(x.opened_at).values.astype("datetime64[ms]").astype("int64")
        why = x.close_reason.map({"STOP_LOSS": "STOP", "FRENZY_TP": "TP"}).fillna("OTHER").values
        out["BATCH (6, as booked = today's exit)"] = const(x, x.net.values, why, x.hold_h.values, x.mod_fund_pct.values, te)
    return out


def classify(df, H, H0):
    a, b = df[f"c{H0}_why"].values, df[f"c{H}_why"].values
    ha, hb = df[f"c{H0}_hold"].values, df[f"c{H}_hold"].values
    d = df[f"c{H}_net"].values - df[f"c{H0}_net"].values
    cls = np.full(len(df), "SAME", object)
    same = (a == b) & (np.abs(ha - hb) < 1e-9)
    if H > H0:
        ch = ~same & (a == "TIME")
        cls[ch & (b == "TP")] = "SAVED (time → TP)"
        cls[ch & (b == "STOP")] = "DEEPER (time → stop)"
        cls[ch & np.isin(b, ["TIME", "OPEN"])] = "DRIFT (time later)"
    elif H < H0:
        ch = ~same & np.isin(b, ["TIME", "OPEN"])
        cls[ch & (a == "TP")] = "CUT WINNER (TP → time)"
        cls[ch & (a == "STOP")] = "SAVED LOSS (stop → time)"
        cls[ch & np.isin(a, ["TIME", "OPEN"])] = "EARLY TIME"
        cls[ch & np.isin(a, ["RUNNER", "LATE_TP"])] = "CUT DYN-EXIT (runner / late TP → time)"
    cls[(cls == "SAME") & ~same] = "OTHER"
    return cls, d


def dcr(df, H, x, net_col=None):
    net = df[net_col or f"c{H}_net"].values
    te = df.te.values.astype("int64")
    close = pd.to_datetime(te + (df[f"c{H}_hold"].fillna(0).values * HR).astype("int64"), unit="ms").normalize()
    d0 = pd.to_datetime(te.min(), unit="ms").normalize(); d1 = close.max()
    return C.day_dcr(net, close, d0, d1, x)


def block_rows(s, name, df):
    caps, H0, X, unit = SLEEVES[s]
    rows = []
    day = df.te.values.astype("int64") // DAY
    base = df[f"c{H0}_net"].values
    for H in caps:
        n = df[f"c{H}_net"].values
        f = df[f"c{H}_fund"].values
        hold = df[f"c{H}_hold"].values * (60 if unit == "min" else 1)
        d = n - base
        cls, _ = classify(df, H, H0)
        m, lo, hi, pb = C.cluster_boot(d, day) if H != H0 else (0.0, np.nan, np.nan, np.nan)
        dp = (n - f) - (base - df[f"c{H0}_fund"].values)
        r = dict(sleeve=s, block=name, cap=f"{H}{unit}" + (" (today)" if H == H0 else ""), H=H, N=len(df),
                 WR=100 * (n > 0).mean(), avg=n.mean(), sum_pct=n.sum(), usd=n.sum() / 100 * X * EQ,
                 dcr=dcr(df, H, X)[0], maxdd=dcr(df, H, X)[1], hold_avg=np.nanmean(hold), hold_med=np.nanmedian(hold),
                 fund_avg=np.nanmean(f), time_exits=int(np.isin(df[f"c{H}_why"].values, ["TIME", "OPEN", "MAX_HOLD_TIME"]).sum()),
                 changed=int((cls != "SAME").sum()), d=m, lo=lo, hi=hi, p_better=pb, d_price=dp.mean(), fund_eff=m - dp.mean(),
                 untouched_max=float(np.abs(d[cls == "SAME"]).max()) if (cls == "SAME").any() else 0.0)
        if f"c{H}_n02" in df:
            r["d_s02"] = (df[f"c{H}_n02"].values - base).mean(); r["d_s30"] = (df[f"c{H}_n30"].values - base).mean()
        for k in ("SAVED (time → TP)", "DEEPER (time → stop)", "DRIFT (time later)", "CUT WINNER (TP → time)", "SAVED LOSS (stop → time)",
                  "EARLY TIME", "CUT DYN-EXIT (runner / late TP → time)"):
            mk = cls == k
            if mk.any():
                r[k] = f"{int(mk.sum())} · {d[mk].sum():+.1f}"
        if H != H0 and (d != 0).any():
            lp = pd.Series(d).groupby(df.pair.values).sum(); ld = pd.Series(d).groupby(day).sum(); tot = d.sum()
            r["loo_pair"] = "{:+.3f}…{:+.3f}".format(*[((tot - lp) / (len(d) - pd.Series(1, index=df.pair.values).groupby(level=0).sum()
                                                                          .reindex(lp.index))).agg(fn) for fn in ("min", "max")])
            r["loo_day"] = "{:+.3f}…{:+.3f}".format(*[((tot - ld) / (len(d) - pd.Series(1, index=day).groupby(level=0).sum()
                                                                         .reindex(ld.index))).agg(fn) for fn in ("min", "max")])
            sg = np.sign(tot) if tot != 0 else 1
            r["top_pair"] = f"{lp.idxmax() if sg > 0 else lp.idxmin()} {100 * (lp.max() if sg > 0 else lp.min()) / tot:.0f}%"
            r["top_day"] = f"{100 * (ld.max() if sg > 0 else ld.min()) / tot:.0f}%"
        rows.append(r)
    return rows


def halfyear(s, df):
    caps, H0, _, unit = SLEEVES[s]
    hy = df.te.map(half)
    out = []
    for H in caps:
        if H == H0:
            continue
        d = df[f"c{H}_net"] - df[f"c{H0}_net"]
        g = d.groupby(hy).mean()
        out.append(dict(sleeve=s, cap=f"{H}{unit}", **{k: v for k, v in g.items()}, pos=f"{int((g > 0).sum())}/{int((g != 0).sum())}"))
    return out


def hold_dist(s, df):
    caps, H0, _, unit = SLEEVES[s]
    why = df[f"c{H0}_why"].replace({"MAX_HOLD_TIME": "TIME", "FRENZY_TP": "TP", "STOP_LOSS": "STOP"})
    net = df[f"c{H0}_net"]
    grp = np.where(np.isin(why, ["TIME", "OPEN"]), "time-exit", np.where(net > 0, "winner", "loser"))
    hold = df[f"c{H0}_hold"] * (60 if unit == "min" else 1)
    out = []
    for g in ("winner", "loser", "time-exit"):
        h = hold[grp == g]
        if len(h):
            q = np.percentile(h, [10, 25, 50, 75, 90])
            out.append(dict(sleeve=s, group=g, N=len(h), avg_net=net[grp == g].mean(), avg_fund=df[f"c{H0}_fund"][grp == g].mean(),
                            p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4], max=h.max()))
    return out, hold.values, grp


def conditional(s, df):
    """for fills still open at hour h (today's exit later than h): avg result of HOLDING ON (today's exit) vs closing at h (the cap-h cell)."""
    caps, H0, _, unit = SLEEVES[s]
    out = []
    for H in caps:
        if H >= H0:
            continue
        m = df[f"c{H0}_hold"] > df[f"c{H}_hold"] + 1e-9
        if m.sum() == 0:
            continue
        x = df[m]
        out.append(dict(sleeve=s, open_at=f"{H}{unit}", N=int(m.sum()), share=100 * m.mean(),
                        close_now=x[f"c{H}_net"].mean(), hold_on=x[f"c{H0}_net"].mean(),
                        hold_on_TP=100 * (x[f"c{H0}_why"].isin(["TP", "FRENZY_TP"])).mean(),
                        hold_on_STOP=100 * (x[f"c{H0}_why"].isin(["STOP", "STOP_LOSS"])).mean(),
                        extra_funding=(x[f"c{H0}_fund"] - x[f"c{H}_fund"]).mean()))
    return out


def md(df, fl=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:+,.0f}" if abs(v) >= 1000 else f"{v:+.{fl}f}"))
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    out += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    rows, hys, hds, conds, charts, extra = [], [], [], [], {}, []
    for s in SLEEVES:
        bt = load_bt(s)
        blocks = load_real(s)
        blocks.update({"BACKTEST DESIGN (<2025-07)": bt[bt.period == "DESIGN"], "BACKTEST HOLDOUT (≥2025-07)": bt[bt.period == "HOLDOUT"],
                       "BACKTEST ALL": bt})
        for nm, df in blocks.items():
            if len(df):
                rows += block_rows(s, nm, df.reset_index(drop=True))
        hys += halfyear(s, bt)
        hd, hold, grp = hold_dist(s, bt); hds += hd
        conds += conditional(s, bt)
        charts[s] = (bt, hold, grp)
        if s in ("EXH", "DIST", "FB"):
            # funding grows with hold: funding per fill by today's hold bucket
            caps, H0, _, _ = SLEEVES[s]
            edges = [0, 1, 4, 8, 12, 16, 24, 36, 48, 72, 96, 121]
            b = pd.cut(bt[f"c{H0}_hold"], [e for e in edges if e <= H0 + 1] + ([H0 + 1] if H0 + 1 not in edges else []))
            g = bt.groupby(b, observed=True).agg(N=("pair", "size"), avg_net=(f"c{H0}_net", "mean"), avg_fund=(f"c{H0}_fund", "mean"))
            g["sleeve"] = s; extra.append(g.reset_index().rename(columns={f"c{H0}_hold": "hold_h"}))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(D, "summary.csv"), index=False)
    L = ["# MAX_HOLD_STUDY 2026-11-21 — full tables (auto-generated by scripts/MAX_HOLD_STUDY_2026-11-21_report.py)", "",
         "Units: % of notional per fill at ONE size, net of fees, slip and real funding. Δ = cap − today per fill (all fills; untouched fills "
         "Δ = 0, `untouched_max` must read 0). CI = day-block bootstrap 95 % (entry day clusters, 4,000 draws). $ = Σ % × X × $10,000 "
         "(X = 0.61 EXH / DIST / FB, 0.305 HT / WILLY). DCR = one-size daily compound over every calendar day of the block, %/day. "
         "fund_eff = Δ − Δ price-only (the part of Δ due to funding). d_s02 / d_s30 = Δ with the time-exit slip at 0.02 / 0.30 %. "
         "Change classes: 'n · Σ Δ pts'. Backtest rows whose funding months are missing for the longest cap are dropped (EXH 80, DIST 2, FB 6).", ""]
    cols = ["block", "cap", "N", "WR", "avg", "usd", "dcr", "maxdd", "hold_avg", "hold_med", "fund_avg", "time_exits", "changed", "d", "lo",
            "hi", "p_better", "fund_eff", "d_s02", "d_s30", "untouched_max"]
    cl2 = ["block", "cap", "SAVED (time → TP)", "DEEPER (time → stop)", "DRIFT (time later)", "CUT WINNER (TP → time)",
           "SAVED LOSS (stop → time)", "EARLY TIME", "CUT DYN-EXIT (runner / late TP → time)", "loo_pair", "loo_day", "top_pair", "top_day"]
    for s in SLEEVES:
        x = T[T.sleeve == s]
        L += [f"## {s}", "", "### Before → after per cap (three blocks)", "", md(x[[c for c in cols if c in x]]), "",
              "### Change split, LOO, concentration", "", md(x[[c for c in cl2 if c in x]].dropna(how="all", axis=1)), ""]
        h = pd.DataFrame([r for r in hys if r["sleeve"] == s]).drop(columns="sleeve")
        L += ["### Δ per half-year (backtest, % per fill over ALL fills of the half)", "", md(h), ""]
        L += ["### Hold-time distribution under today's exit (backtest)", "", md(pd.DataFrame([r for r in hds if r["sleeve"] == s]).drop(columns="sleeve"), 2), ""]
        cc = [r for r in conds if r["sleeve"] == s]
        if cc:
            L += ["### Fills still open at the cap: close now vs hold on (backtest)", "", md(pd.DataFrame(cc).drop(columns="sleeve"), 2), ""]
        ex = [e for e in extra if e.sleeve.iloc[0] == s]
        if ex:
            L += ["### Funding by today's hold (backtest)", "", md(ex[0].drop(columns="sleeve"), 3), ""]
    open(os.path.join(RP, "MAX_HOLD_STUDY_2026-11-21_tables.md"), "w").write("\n".join(L))
    chart(T, charts)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(T[["sleeve", "block", "cap", "N", "WR", "avg", "dcr", "hold_avg", "fund_avg", "time_exits", "changed", "d", "lo", "hi", "fund_eff"]]
          .round(3).to_string())


def chart(T, charts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(2, 5, figsize=(22, 8.5), gridspec_kw=dict(height_ratios=[1.25, 1]))
    fig.patch.set_facecolor("#fcfcfb")
    names = {"EXH": "EXHAUSTION_SHORT", "DIST": "DISTRIBUTION_SHORT", "FB": "FLATBASE_SHORT", "HT": "FRENZY_WILLY_HT", "WILLY": "FRENZY_WILLY"}
    for j, s in enumerate(SLEEVES):
        caps, H0, _, unit = SLEEVES[s]
        a = ax[0, j]; a.set_facecolor("#fcfcfb")
        for blk, col, lab in (("BACKTEST ALL", BLUE, "backtest ALL"), ("BACKTEST HOLDOUT (≥2025-07)", ORANGE, "backtest HOLDOUT")):
            x = T[(T.sleeve == s) & (T.block == blk)].sort_values("H")
            lo = np.where(x.H == H0, 0, x.lo); hi = np.where(x.H == H0, 0, x.hi)
            a.fill_between(x.H, lo, hi, color=col, alpha=0.15, lw=0)
            a.plot(x.H, x.d, color=col, lw=2, marker="o", ms=5, label=lab + " (95 % day-block CI)")
        real = T[(T.sleeve == s) & ~T.block.str.startswith("BACKTEST")]
        for k, (blk, g) in enumerate(real.groupby("block")):
            g = g.sort_values("H")
            a.plot(g.H, g.d, color=AQUA, lw=0, marker="s^D"[k % 3], ms=7, label=f"real: {blk.split(' (')[0].lower()} (N {int(g.N.iloc[0])})")
        a.axhline(0, color=MUTED, lw=0.8); a.axvline(H0, color=MUTED, lw=1, ls="--")
        a.text(H0, a.get_ylim()[1], " today", color=MUTED, va="top", fontsize=9)
        a.set_title(names[s], fontsize=11, color=INK, loc="left")
        a.set_xlabel(f"max hold ({unit})", color=MUTED); a.set_xticks(caps)
        if j == 0:
            a.set_ylabel("Δ vs today, % per fill (net of funding)", color=MUTED)
        a.legend(fontsize=7.5, frameon=False, loc="lower right")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        # hold distribution (backtest, today's exit)
        b = ax[1, j]; b.set_facecolor("#fcfcfb")
        bt, hold, grp = charts[s]
        top = H0 * 1.02
        bins = np.linspace(0, top, 25)
        for g_, col in (("winner", BLUE), ("loser", ORANGE), ("time-exit", AQUA)):
            h = hold[grp == g_]
            if len(h):
                b.hist(h, bins=bins, color=col, alpha=0.55, label=f"{g_} ({len(h)})", edgecolor="#fcfcfb", lw=0.5)
        b.set_yscale("log"); b.set_xlabel(f"hold under today's exit ({unit}), backtest", color=MUTED)
        if j == 0:
            b.set_ylabel("fills (log)", color=MUTED)
        b.legend(fontsize=8, frameon=False)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)
    fig.suptitle("Max-hold study 2026-11-21 — Δ per fill vs today's cap (top) · hold-time distribution by outcome (bottom); funding charged, "
                 "one size", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(RP, "MAX_HOLD_STUDY_2026-11-21_chart.png"), dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
