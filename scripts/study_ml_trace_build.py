#!/usr/bin/env python3
"""MOMENTUM-LONG recall trace (2026-10-08) — step 1: the matched set (live ML fills ↔ yr5 replay ML fills).

Read-only. Template: scripts/study_fade_trace_build.py (SPIKE_FADE recall trace).
LIVE  = every momentum-long fill live actually took, AS TRADED, full-size (probes + MANUAL out), CLOSED, deduped on
        (opened_at, pair, direction): BASE from the COMBINED raw pool (the master's BASE is pre-screened), B1 from BATCH1_FINAL
        (the master's B1 = the Jul-31 ANCHOR screen), B2.. from reports/MASTER_POOL_stacked.csv (all rows; stack_keep carried),
        cross-checked against every archived batch CSV and the newest Downloads export. Window = the yr5 replay [Jan-04, Oct-04).
REPLAY = yr5 (code 181131e, today's frozen config, synthetic live cadence, 3 seeds) MOM-long non-probe fills, warm-up trimmed
        (scripts/yr5_fills_trimmed.py).
Match = same pair, LONG, |Δopen| ≤ 10 min, any seed (nearest per seed).
Also joined per live fill: the Oct-4 audit's AS-WAS replay at live's scan clock (reports/ML_AUDIT_after_trades.csv: as-was config)
and TODAY's-config replay at live's scan clock (reports/ML_AUDIT_today_trades.csv) → which of the three views reproduce it.
Live-up periods = union of live batch spans (BASE from the COMBINED pool) merged across sub-hour seams.
Writes reports/study_ml_trace_matchset.csv (one row per live ML fill and per replay ML fill).
Usage: venv/bin/python scripts/study_ml_trace_build.py
"""
import glob, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yr5_fills_trimmed as YT                                          # noqa: E402

REP = os.path.join(ROOT, "reports")
W0, W1 = pd.Timestamp("2026-01-04"), pd.Timestamp("2026-10-04")
TOL = pd.Timedelta(minutes=10)


def ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace("T", " "), format="mixed", errors="coerce")


def batch_files():
    fs = [os.path.join(REP, "COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv"),
          os.path.join(REP, "BATCH1_2026-07-11to31_orders_FINAL.csv")] + sorted(glob.glob(os.path.join(REP, "BASELINE*_*.csv")))
    return [f for f in fs if "ANCHOR" not in f and "superseded" not in f and f.endswith(".csv")]


def live_up_intervals():
    iv = []
    for f in batch_files():
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "closed_at"))
        o = ts(d.opened_at)
        if "COMBINED" in f:          # BASE span only (later rows of the COMBINED pool are B1 duplicates)
            o = o[o < pd.Timestamp("2026-07-11")]
        a = o.min(); z = o.max()
        if "COMBINED" not in f:
            z = max(z, ts(d.closed_at).max())
        iv.append((a, z))
    iv.sort()
    out = []
    for a, z in iv:
        if out and a <= out[-1][1] + pd.Timedelta(hours=1):
            out[-1] = (out[-1][0], max(out[-1][1], z))
        else:
            out.append((a, z))
    return out


def is_up(t, iv):
    t = pd.Series(t)
    m = np.zeros(len(t), bool)
    for a, z in iv:
        m |= ((t >= a) & (t <= z)).values
    return m


def is_ml(d):
    return (d.entry_strategy.fillna("MOMENTUM").astype(str) == "MOMENTUM") & (d.direction == "LONG")


def is_probe(d):
    p = d.cell_multiplier_source.fillna("").astype(str).str.contains("PROBE")
    if "is_probe" in d:
        p |= d.is_probe.astype(str).str.lower().isin(["true", "1", "1.0"])
    return p


def load_live():
    P = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
    C = pd.read_csv(os.path.join(REP, "COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv"), low_memory=False)
    B1 = pd.read_csv(os.path.join(REP, "BATCH1_2026-07-11to31_orders_FINAL.csv"), low_memory=False)
    base = C[C.opened_at.astype(str) < "2026-07-11"].assign(era="BASE")
    L = pd.concat([base, B1.assign(era="B1"), P[~P.era.isin(["BASE", "B1"])]], ignore_index=True)
    L = L[(L.status == "CLOSED") & is_ml(L)].copy()
    L["oa"] = L.opened_at.astype(str).str[:19].str.replace(" ", "T")
    L = L.drop_duplicates(["oa", "pair", "direction"])
    L["t"] = ts(L.opened_at).values
    L = L[(L.t >= W0) & (L.t < W1)]
    L["probe"] = is_probe(L)
    # master join (stack_keep, stack_reason, CF stack %)
    P["oa"] = P.opened_at.astype(str).str[:19].str.replace(" ", "T")
    pk = P.drop_duplicates(["oa", "pair", "direction"]).set_index(["oa", "pair", "direction"])
    keys = list(zip(L.oa, L.pair, L.direction))
    L["in_master"] = [k in pk.index for k in keys]
    for c in ("stack_keep", "stack_block_reason", "stack_pct"):
        L["m_" + c] = [pk[c].get(k, np.nan) if (k in pk.index and c in pk) else np.nan for k in keys]
    L["pct_raw"] = pd.to_numeric(L.pnl_percentage, errors="coerce")
    sp = pd.to_numeric(L.m_stack_pct, errors="coerce")             # master: P&L % under today's rules (kept rows; path CFs applied)
    L["pct_stack"] = np.where(sp.notna(), sp, L.pct_raw)
    # cross-check vs every batch CSV + newest export
    seen = set(zip(L.oa, L.pair))
    extra = []
    dl = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime)[-1:]
    for f in batch_files()[2:] + dl:
        d = pd.read_csv(f, low_memory=False)
        if "entry_strategy" not in d:
            continue
        d = d[(d.status == "CLOSED") & is_ml(d) & ~is_probe(d)]
        for oa, p in zip(d.opened_at.astype(str).str[:19].str.replace(" ", "T"), d.pair):
            if W0 <= pd.Timestamp(oa) < W1 and (oa, p) not in seen:
                extra.append((os.path.basename(f), oa, p))
    return L, extra


def audit_join(L, name):
    f = os.path.join(REP, f"ML_AUDIT_{name}_trades.csv")
    A = pd.read_csv(f)
    A = A[A.kind != "EXTRA"].copy()
    A["oa"] = pd.to_datetime(A.live_open).dt.strftime("%Y-%m-%dT%H:%M:%S")
    A = A.drop_duplicates(["oa", "pair"]).set_index(["oa", "pair"])
    k = list(zip(L.oa, L.pair))
    L[f"{name}_kind"] = [A.kind.get(x, "") for x in k]
    L[f"{name}_cls"] = [A.cls.get(x, "") for x in k]
    L[f"{name}_bt_pct"] = [A.bt_pct.get(x, np.nan) for x in k]
    return L


def main():
    iv = live_up_intervals()
    L, extra = load_live()
    L = audit_join(L, "after"); L = audit_join(L, "today")
    R = YT.load(sleeves=["MOM-long"])
    R = R[(R.t >= W0) & (R.t < W1)].copy()
    LF = L[~L.probe].copy()
    LF["m_seeds"] = ""; LF["m_dt_s"] = np.nan; LF["pct_rep"] = np.nan; LF["rep_reason"] = ""; LF["rep_entry_diff"] = np.nan
    R["m_live_idx"] = -1
    for i, r in LF.iterrows():
        c = R[(R.pair == r.pair) & ((R.t - r.t).abs() <= TOL)]
        if not len(c):
            continue
        c = c.assign(adt=(c.t - r.t).abs()).sort_values("adt").drop_duplicates("seed")
        LF.at[i, "m_seeds"] = ",".join(str(s) for s in sorted(c.seed))
        LF.at[i, "m_dt_s"] = (c.t - r.t).dt.total_seconds().median()
        LF.at[i, "pct_rep"] = c.pct.mean()
        LF.at[i, "rep_reason"] = "|".join(sorted(set(c.close_reason.astype(str).str.split(" L").str[0])))
        LF.at[i, "rep_entry_diff"] = ((pd.to_numeric(c.entry_price) / float(r.entry_price) - 1) * 100).mean()
        R.loc[c.index, "m_live_idx"] = i
    LF["n_seeds"] = LF.m_seeds.str.count(",") + (LF.m_seeds != "").astype(int)
    R["live_up"] = is_up(R.t, iv)
    out_l = pd.DataFrame({"side": "LIVE", "era": LF.era.values, "pair": LF.pair.values, "t": LF.t.values, "seed": 0,
                          "in_master": LF.in_master.values, "stack_keep": LF.m_stack_keep.values, "stack_reason": LF.m_stack_block_reason.values,
                          "pct_live_raw": LF.pct_raw.values, "pct_live_stack": LF.pct_stack.values,
                          "close_reason": LF.close_reason.astype(str).str.split(" L").str[0].values, "cell": LF.cell_multiplier_source.values,
                          "entry_order_type": LF.get("entry_order_type", pd.Series([""] * len(LF))).values,
                          "entry_price": pd.to_numeric(LF.entry_price, errors="coerce").values,
                          "peak_pnl": pd.to_numeric(LF.get("peak_pnl"), errors="coerce").values,
                          "n_seeds": LF.n_seeds.values, "m_seeds": LF.m_seeds.values, "m_dt_s": LF.m_dt_s.values,
                          "pct_rep": LF.pct_rep.values, "rep_reason": LF.rep_reason.values, "rep_entry_diff": LF.rep_entry_diff.values,
                          "asw_kind": LF.after_kind.values, "asw_cls": LF.after_cls.values, "asw_bt_pct": LF.after_bt_pct.values,
                          "lp_kind": LF.today_kind.values, "lp_cls": LF.today_cls.values, "live_up": True})
    for c in [c for c in LF.columns if c.startswith("entry_")]:
        out_l[c] = LF[c].values
    out_r = pd.DataFrame({"side": "REPLAY", "pair": R.pair.values, "t": R.t.values, "seed": R.seed.values, "tag": R.tag.values,
                          "pct_rep": R.pct.values, "rep_reason": R.close_reason.astype(str).str.split(" L").str[0].values,
                          "cell": R.cell_multiplier_source.values, "entry_price": pd.to_numeric(R.entry_price, errors="coerce").values,
                          "peak_pnl": pd.to_numeric(R.get("peak_pnl"), errors="coerce").values,
                          "entry_order_type": R.get("entry_order_type", pd.Series([""] * len(R))).values,
                          "matched_live": R.m_live_idx.values >= 0, "live_up": R.live_up.values})
    for c in [c for c in R.columns if c.startswith("entry_")]:
        out_r[c] = R[c].values
    out = pd.concat([out_l, out_r], ignore_index=True)
    out.to_csv(os.path.join(REP, "study_ml_trace_matchset.csv"), index=False)
    pd.DataFrame(iv, columns=["up_from", "up_to"]).to_csv(os.path.join(REP, "study_ml_trace_liveup.csv"), index=False)
    # console
    print("live-up intervals:", [(str(a)[:16], str(z)[:16]) for a, z in iv])
    print(f"live ML fills (window) {len(L)} · full-size {len(LF)} · probes {int(L.probe.sum())} · in master {int(LF.in_master.sum())} "
          f"· kept {int((LF.m_stack_keep == True).sum())} · extra batch/export rows not in the live set: {len(extra)} {extra[:6]}")
    rep = LF.n_seeds > 0
    print(f"reproduced by ≥1 yr5 seed: {int(rep.sum())}/{len(LF)} = {rep.mean() * 100:.0f}% · per seed "
          + " / ".join(f"s{s} {LF.m_seeds.str.contains(str(s)).mean() * 100:.0f}%" for s in (1, 2, 3)))
    for lab, m in (("kept", LF.m_stack_keep == True), ("removed", LF.m_stack_keep == False), ("not in master", ~LF.in_master)):
        g = LF[m]
        print(f"  {lab:14s} N {len(g):3d} live {g.pct_raw.mean():+.3f} · ≥1 seed {int((g.n_seeds > 0).sum())} ({(g.n_seeds > 0).mean() * 100:.0f}%)"
              f" · asw {int((g.after_kind == 'MATCHED').sum())} · lp {int((g.today_kind == 'MATCHED').sum())}")
    up = R[R.live_up]
    print(f"replay ML fills {len(R)} ({R.seed.value_counts().sort_index().to_dict()}) · live-up {len(up)} ({len(up) / 3:.1f}/seed, avg {up.pct.mean():+.3f}) "
          f"· matched {int((R.m_live_idx >= 0).sum())} · live-down {int((~R.live_up).sum())} (avg {R[~R.live_up].pct.mean():+.3f})")


if __name__ == "__main__":
    main()
