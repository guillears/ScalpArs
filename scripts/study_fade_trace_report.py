#!/usr/bin/env python3
"""SPIKE_FADE recall trace (2026-10-08) — step 4: classify every signal, quantify each class, calibrated fade expectancy.

Read-only over reports/study_fade_trace_signals_raw.csv (step 3), the live batch CSVs (live book / pair held at a replay-only
signal) and the yr5 fade fills (year-wide decision-source split). Writes reports/study_fade_trace_signals.csv (one row per signal:
live fades at fade level, replay fades per seed, with final class + family) and prints every table used in
reports/SPIKE_FADE_RECALL_TRACE_2026-10-08.md.
Usage: venv/bin/python scripts/study_fade_trace_report.py
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yr5_fills_trimmed as YT                                   # noqa: E402
from study_fade_trace_build import batch_files, ts               # noqa: E402

REP = os.path.join(ROOT, "reports")
C = os.path.join(REP, "backtest_cache")
SCAN_S = 111.6
FLOOR = 2_000_000.0
FB_LIVE_FROM = pd.Timestamp("2026-08-10 13:48:19")              # FADE_FRESHBREAK went live (config history)
REALMONEY = (pd.Timestamp("2026-08-24 21:02"), pd.Timestamp("2026-08-27 16:35"))   # B4/B5 real-money batches (is_paper False)
MATCH_GAP = 0.177 - 0.132                                        # same-signal pricing gap (live − replay) on the 78 matched rows

FAMILY = {
    "MATCHED": "matched",
    "DATA_STALE_CANDLE": "A data: stale 1-min candle view",
    "CLOSED_ONLY_VIEW": "A data: stale 1-min candle view",
    "UNIVERSE_VOL_FLOOR": "B data: 24h-volume lag (scanner $2M floor)",
    "SCAN_PHASE": "C timing: scan instant / phase",
    "SCAN_INSTANT_EDGE": "C timing: scan instant / phase",
    "FIRED_OTHER_TIME": "C timing: scan instant / phase",
    "GATE_EDGE": "C timing: scan instant / phase",
    "GATE_INPUT_DIFF": "D gate read at a different instant",
    "TODAY_RULE_BLOCKS": "E rules: today's rules block it",
    "UNIVERSE_BLACKLIST": "E rules: today's rules block it",
    "CAPACITY_REPLAY_HELD_PAIR": "F capacity (replay held the pair)",
    "TRIGGER_PASS_NO_OPEN": "G unexplained",
}


def boot(v, d, w=None, B=4000, seed=7):
    """day-clustered bootstrap 95 % CI of the (weighted) mean."""
    v = np.asarray(v, float); d = np.asarray(d); w = np.ones(len(v)) if w is None else np.asarray(w, float)
    if len(v) < 3:
        return (np.nan, np.nan)
    g = pd.DataFrame({"s": v * w, "w": w, "d": d}).groupby("d")[["s", "w"]].sum()
    s, c, k = g.s.values, g.w.values, len(g)
    rng = np.random.default_rng(seed); i = rng.integers(0, k, (B, k))
    m = s[i].sum(1) / c[i].sum(1)
    return tuple(float(x) for x in np.round(np.percentile(m, [2.5, 97.5]), 3))


def refine_live(r):
    """seed-level class → refined class (step-3 class + checks that need the fade's context)."""
    c = r["class"]
    if c == "MATCHED":
        return c
    if r.r_vol24 == r.r_vol24 and r.r_vol24 < FLOOR:
        return "UNIVERSE_VOL_FLOOR"
    if c == "UNIVERSE_BLACKLIST":
        return c
    if c.startswith("GATE_FRESHBREAK") and r.t < FB_LIVE_FROM:
        return "TODAY_RULE_BLOCKS"          # the gate did not exist live then; master stamps cannot recompute it → kept by mistake
    if c.endswith("@EDGE"):
        return "GATE_EDGE"
    if c.startswith("GATE_"):
        return "GATE_INPUT_DIFF"
    if c == "DATA_STALE_CANDLE" and "CLOSED_ONLY" in str(r.scan_srcs):
        return "CLOSED_ONLY_VIEW"
    return c


def live_book():
    rows = []
    for f in batch_files():
        d = pd.read_csv(f, low_memory=False)
        es = d.entry_strategy.fillna("MOMENTUM").astype(str) if "entry_strategy" in d else pd.Series(["MOMENTUM"] * len(d))
        d = d[es != "MANUAL"]
        rows.append(pd.DataFrame({"pair": d.pair, "a": ts(d.opened_at).values, "z": ts(d.closed_at).values}))
    B = pd.concat(rows).dropna(subset=["a"]).drop_duplicates(["pair", "a"])
    B["z"] = B.z.fillna(pd.Timestamp("2026-10-09"))
    return B


def main():
    R0 = pd.read_csv(os.path.join(REP, "study_fade_trace_signals_raw.csv"), low_memory=False)
    R0["t"] = pd.to_datetime(R0.t)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    # ───────────── LIVE side ─────────────
    L = R0[R0.side == "LIVE"].copy()
    L["kept"] = L.stack_keep.astype(str).str.lower() == "true"
    L["rclass"] = [refine_live(r) for _, r in L.iterrows()]
    L.loc[~L.kept & (L.rclass != "MATCHED"), "rclass"] = "TODAY_RULE_BLOCKS"
    L["family"] = L.rclass.map(FAMILY).fillna("G unexplained")
    order = list(dict.fromkeys(FAMILY.values()))

    def fade_row(g):
        fams = g.family.value_counts()
        top = fams[fams == fams.max()].index.tolist()
        fam = sorted(top, key=lambda f: order.index(f) if f in order else 99)[0]
        if g.n_seeds.iloc[0] > 0:
            fam = "matched"
        return pd.Series({"family": fam, "seed_classes": "|".join(g.rclass.astype(str)), "kept": g.kept.iloc[0], "era": g.era.iloc[0],
                          "n_seeds": g.n_seeds.iloc[0], "pct_live_raw": g.pct_live_raw.iloc[0], "pct_live_stack": g.pct_live_stack.iloc[0],
                          "pct_rep_matched": g.pct_rep_matched.iloc[0], "live_truth_trig": g.live_truth_trig.iloc[0],
                          "live_rep_src": g.live_rep_src.iloc[0], "trig_window_s": g.trig_window_s.iloc[0], "bits_all": g.bits_all.iloc[0]})

    LF = L.groupby(["pair", "t"], sort=False).apply(fade_row).reset_index()
    LF["day"] = LF.t.dt.date
    print("=== LIVE fades in the replay window (Jul-28 → Oct-4), fade level ===")
    print("trigger reproduced on REAL ticks at the live decision instant:", f"{LF.live_truth_trig.mean():.1%}",
          "| replay decision view at the live instant:", LF.live_rep_src.astype(str).str.split("(").str[0].value_counts().to_dict())
    tab = LF.groupby(["kept", "family"]).agg(N=("pct_live_raw", "size"), WR=("pct_live_raw", lambda x: round((x > 0).mean() * 100)),
                                              avg_raw=("pct_live_raw", "mean"), avg_stack=("pct_live_stack", "mean"),
                                              win_s=("trig_window_s", "median")).round(3)
    print(tab)
    print("\nseed-level refined classes, live KEPT fades not reproduced in that seed:")
    sl = L[L.kept & (L.rclass != "MATCHED")]
    print(sl.groupby("rclass").agg(seed_rows=("pct_live_raw", "size"), avg_live=("pct_live_raw", "mean")).round(3).sort_values("seed_rows", ascending=False))
    print("\ntrigger-window width (s, real ticks, kept fades) by family:")
    print(LF[LF.kept].groupby("family").trig_window_s.describe()[["count", "25%", "50%", "75%"]])

    # ───────────── REPLAY side ─────────────
    Rp = R0[R0.side == "REPLAY"].copy()
    B = live_book()
    st = []
    for p, t in zip(Rp.pair, Rp.t):
        o = B[(B.a <= t) & (B.z > t)]
        st.append((len(o), bool((o.pair == p).any())))
    Rp["live_open_n"] = [s[0] for s in st]; Rp["live_pair_held"] = [s[1] for s in st]
    Rp["src0"] = Rp.src.astype(str).str.split("(").str[0]

    def rclass(r):
        if str(r.matched_live) == "True":
            return "MATCHED"
        if str(r.live_up) == "False":
            return "LIVE_OFFLINE"
        if REALMONEY[0] <= r.t <= REALMONEY[1]:
            return "LIVE_REALMONEY_ERA"
        if not r.parity_rep_trig:
            return "PARITY_FAIL"
        if r.gate_live_era not in ("PASS", "NA"):
            return f"ERA_CONFIG_{r.gate_live_era}"
        if (not r.truth_trig_pm20) and r.src0 in ("1M", "CLOSED_ONLY"):
            return "REPLAY_STALE_ARTEFACT"
        if r.live_pair_held:
            return "LIVE_PAIR_HELD"
        if r.live_open_n >= 10:
            return "LIVE_BOOK_FULL"
        if r.lj_scans == r.lj_scans and r.lj_scans > 0:
            return "LIVE_SCAN_PHASE" if not r.lj_truth_trig else "LIVE_TRIGGERED_NOT_TAKEN"
        if r.trig_window_s == r.trig_window_s and r.trig_window_s < SCAN_S:
            return "LIVE_SCAN_PHASE"
        if not r.truth_trig_pm20:
            return "TRUTH_NO_TRIGGER"
        return "UNEXPLAINED"

    Rp["class"] = [rclass(r) for _, r in Rp.iterrows()]
    ns = Rp.seed.nunique()
    print("\n=== REPLAY fades in the window (3 seeds) ===")
    print("parity — the replay's own fills re-trigger in this reconstruction at their decision scan:", f"{Rp.parity_rep_trig.mean():.1%}")
    for a, b in (("stamp_vol24", "calc_vol24"), ("stamp_ndi", "calc_ndi"), ("stamp_btc4h", "calc_btc4h")):
        x = Rp[[a, b]].astype(float).dropna()
        print(f"stamp parity {a} vs {b}: N {len(x)}, median |Δ| {np.abs(x[a] - x[b]).median():.3g}, within 0.1 % "
              f"{(np.abs(x[a] - x[b]) <= 1e-3 * np.maximum(1, np.abs(x[a]))).mean():.1%}")
    t = Rp.groupby("class").agg(N=("pct_rep", "size"), N_seed=("pct_rep", lambda x: round(len(x) / ns, 1)),
                                WR=("pct_rep", lambda x: round((x > 0).mean() * 100)), avg=("pct_rep", "mean"),
                                src_1M=("src0", lambda x: round((x != "TICKS").mean() * 100)),
                                truth_trig=("truth_trig_pm20", lambda x: round(x.mean() * 100)), win_s=("trig_window_s", "median")).round(3)
    print(t.sort_values("N", ascending=False))

    # ───────────── decomposition of the live-vs-replay gap (live-up window) ─────────────
    W = Rp[Rp.live_up.astype(str) == "True"].copy()
    W["day"] = W.t.dt.date
    base = W.pct_rep.mean()
    print(f"\n=== gap decomposition (live-up window) ===\nreplay fades while live was up: {len(W) / ns:.1f}/seed · avg {base:+.3f} · CI {boot(W.pct_rep, W.day)}")
    LK = LF[LF.kept]
    print(f"live kept fades: N {len(LK)} · avg raw {LK.pct_live_raw.mean():+.3f} CI {boot(LK.pct_live_raw, LK.day)} · stack {LK.pct_live_stack.mean():+.3f}")
    LK2 = LK[~LK.family.str.startswith("E rules")]
    print(f"live kept fades that today's rules really keep: N {len(LK2)} · avg raw {LK2.pct_live_raw.mean():+.3f} CI {boot(LK2.pct_live_raw, LK2.day)}")
    for c in t.index:
        g = W[W["class"] == c]
        if len(g):
            print(f"  replay {c:26s} {len(g) / ns:5.1f}/seed avg {g.pct_rep.mean():+.3f} · exp without it {W[W['class'] != c].pct_rep.mean():+.3f}")

    # ───────────── calibrated fade expectancy ─────────────
    # harness fix = real-tick forming candle + live-parity 24 h volume. Recovered: live KEPT fades missed for family A/B in a seed,
    # priced at live % − the same-signal gap (the replay prices a shared signal 0.045 below live). Removed: replay fills that fire
    # only on the stale view (the real tape does not trigger at that instant).
    fix_rows = L[L.kept & L.family.str.startswith(("A data", "B data"))]
    rec = fix_rows.groupby(["pair", "t"]).agg(w=("seed", "size"), pct=("pct_live_raw", "first")).reset_index()
    rec["w"] = rec.w / ns; rec["pct"] = rec.pct - MATCH_GAP; rec["day"] = rec.t.dt.date
    keep = W[W["class"] != "REPLAY_STALE_ARTEFACT"]
    mix_v = np.r_[keep.pct_rep.values, rec.pct.values]
    mix_w = np.r_[np.full(len(keep), 1 / ns), rec.w.values]
    mix_d = np.r_[keep.day.values, rec.day.values]
    cal = np.average(mix_v, weights=mix_w)
    print(f"\n(1) live-up window: replay as-is {base:+.3f} → fixed-harness estimate {cal:+.3f} CI {boot(mix_v, mix_d, mix_w)} "
          f"(recovered {rec.w.sum():.1f} fades/seed at {np.average(rec.pct, weights=rec.w):+.3f}; dropped "
          f"{(W['class'] == 'REPLAY_STALE_ARTEFACT').sum() / ns:.1f}/seed stale-view artefacts)")
    F = YT.load(sleeves=["Spike-Fade"])
    rs = {}
    for tag, ch, seed, a, z, w in YT.chunk_windows("yr5"):
        m = json.load(open(f"{C}/replay/{tag}_meta.json"))
        rs[tag] = os.path.getmtime(f"{C}/replay/{tag}_orders.csv") - m["elapsed_s"] - 600
    F["src"] = ["TICKS" if os.path.exists(fp := f"{C}/ticks_q/{p}/{t.strftime('%Y-%m-%d')}.npz") and os.path.getmtime(fp) < rs[tg] else "1M"
                for p, t, tg in zip(F.pair, F.t, F.tag)]
    F["half"] = np.where(F.t < pd.Timestamp("2026-05-20"), "H1", "H2"); F["day"] = F.t.dt.date
    F["stop"] = F.close_reason.astype(str).str.startswith("STOP")
    print("\n=== yr5 fades, whole year, by the forming-candle source the replay decided on ===")
    g = F.groupby(["half", "src"]).agg(N=("pct", "size"), avg=("pct", "mean"), WR=("pct", lambda x: (x > 0).mean()), stop=("stop", "mean"))
    g["ci"] = [boot(F[(F.half == h) & (F.src == s)].pct, F[(F.half == h) & (F.src == s)].day) for h, s in g.index]
    print(g.round(3))
    w1 = W.src0.ne("TICKS").mean()
    share = F.src.eq("1M").mean()
    fy = F.pct.mean()
    fy_cal = fy + (cal - base) * (share / w1 if w1 > 0 else 1.0)
    print(f"FY replay {fy:+.3f} CI {boot(F.pct, F.day)} · 1M-decided share FY {share:.0%} vs live-up window {w1:.0%} "
          f"→ FY calibrated ≈ {fy_cal:+.3f} (window Δ {(cal - base):+.3f} scaled by the 1M share)")
    T = F[F.src == "TICKS"]
    print(f"FY TICKS-decided fills only {T.pct.mean():+.3f} CI {boot(T.pct, T.day)} (N {len(T) / 3:.0f}/seed)")

    out = pd.concat([LF.assign(side="LIVE").rename(columns={"family": "class_family"}),
                     Rp.assign(side="REPLAY", class_family=Rp["class"])], ignore_index=True, sort=False)
    out.to_csv(os.path.join(REP, "study_fade_trace_signals.csv"), index=False)
    print("wrote reports/study_fade_trace_signals.csv")


if __name__ == "__main__":
    main()
