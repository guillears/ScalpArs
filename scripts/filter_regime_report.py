#!/usr/bin/env python3
"""FILTER x BTC-REGIME MATRIX — step 4: gate inventory (priced), survivor / near-miss / gap-screen tables, HEAT focus, real-fill
cross-check (master stack blocks + scout forward re-priced refusals). Writes the tables to $S/frm_tables.md for the report.
Read-only research."""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path[:0] = [ROOT, "scripts"]
S = os.environ.get("S", "/tmp")
import filter_regime_matrix as FM   # noqa: E402  (tags + cut-points)
DAY = 86_400_000
P = pd.read_csv("reports/FILTER_REGIME_MATRIX_signals_priced.csv")
P = P[(P.t >= FM.Y0) & (P.t < FM.Y1) & P.pct.notna()].copy(); P["w"] = P.n_seeds / 3; P["day"] = P.t // DAY
MX = pd.read_csv(FM.OUTCSV)
L = []


def boot_ci(pct, w, day, reps=2000, rng=np.random.default_rng(7)):
    d = pd.DataFrame(dict(s=pct * w, w=w, day=day)).groupby("day").sum()
    PW = rng.poisson(1.0, size=(reps, len(d)))
    b = (PW @ d.s.values) / (PW @ d.w.values)
    return np.nanquantile(b, [.025, .975])


# ── 1. gate inventory, priced ──
rows = []
for g, d in P.groupby("gate"):
    lo, hi = boot_ci(d.pct.values, d.w.values, d.day.values)
    rows.append(dict(gate=g, kind=d.kind.iloc[0], n_all=int(d.n_signals_all.iloc[0]), n_priced=len(d), days=d.day.nunique(),
                     wr=np.average(d.pct > 0, weights=d.w) * 100, avg=np.average(d.pct, weights=d.w), lo=lo, hi=hi))
I = pd.DataFrame(rows).sort_values("avg")
L += ["## G. Gate inventory — every momentum-LONG refusal gate, blocked cohort re-priced (whole year, all states)", "",
      "avg = blocked-cohort avg % per signal (live exit replica, entry t+60 s, n_seeds/3-weighted). Negative = the gate removes losers.",
      "n_all = de-duplicated signals; gates over 8,000 sampled to 8,000 (fixed seed). CI = day-block bootstrap 95 %.", "",
      "| gate | source | signals (all) | priced | days | WR | avg % | 95 % CI |", "|---|---|---|---|---|---|---|---|"]
for r in I.itertuples():
    L.append(f"| {r.gate} | {r.kind} | {r.n_all:,} | {r.n_priced:,} | {r.days} | {r.wr:.0f}% | {r.avg:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] |")
L.append("")
I.to_csv("reports/FILTER_REGIME_MATRIX_2026-10-05_gates.csv", index=False)


def fmt_cell(r, st):
    return (f"{int(r[f'n{st}'])} · {int(r[f'days{st}'])} d · {r[f'avg{st}']:+.3f} [{r[f'ci{st}_lo']:+.3f}, {r[f'ci{st}_hi']:+.3f}] · "
            f"H1 {r[f'h1{st}']:+.3f} ({int(r[f'h1{st}_days'])} d) / H2 {r[f'h2{st}']:+.3f} ({int(r[f'h2{st}_days'])} d)")


def table(df, title, note=""):
    out = [f"## {title}", ""] + ([note, ""] if note else [])
    out += ["| gate | split | state A | A: N · days · avg [CI] · halves | state B | B: N · days · avg [CI] · halves | gap A−B [CI] | verdict |",
            "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        out.append(f"| {r.gate} | {FM.LABEL.get(r.split.split('>')[0].replace('_T', ''), r.split)} ({r.split}) | {r.stateA} | {fmt_cell(r, 'A')} | {r.stateB} | "
                   f"{fmt_cell(r, 'B')} | {r.gap:+.3f} [{r.gap_ci_lo:+.3f}, {r.gap_ci_hi:+.3f}] | {r.verdict} |")
    return out + [""]


surv = MX[MX.verdict.str.startswith("PASS")].copy(); surv["ag"] = surv.gap.abs(); surv = surv.sort_values("ag", ascending=False)
near = MX[MX.verdict.isin(["half h1 flips", "half h2 flips", "gap CI ∋ 0", "days<8", "half h1 thin", "half h2 thin"])].copy()
near["ag"] = near.gap.abs(); near = near.sort_values("ag", ascending=False)
gs = MX[MX.gap_screen].copy(); gs["ag"] = gs.gap.abs(); gs = gs.sort_values("ag", ascending=False)
L += table(surv, "S. Survivors of the full regime-conditional bar (opposite signs + N/days + both halves + gap CI)")
L += table(near, "N. Near misses — opposite signs in the two states but failing one bar (why in the verdict column)")
L += table(gs, "X. Secondary state-gap screen — filter worth significantly more in one state (any signs), gap direction same in both halves",
           "Not the regime-conditional bar (both states may be negative = the filter helps everywhere, more in one). Compare its count with the null.")

# ── HEAT focus ──
L += table(MX[MX.gate == "LONG_HEAT_BLOCK"].sort_values("gap"), "H. LONG_HEAT_BLOCK (yr5 = breadth-only re-scope, bull ≥85) — every split")

# ── real-fill cross-check ──
MP = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
MP = MP[(MP.direction == "LONG") & (MP.status.astype(str) == "CLOSED") & MP.stack_block_reason.notna()]
MP = MP[~MP.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")].drop_duplicates(["opened_at", "pair"])
MP["t"] = (pd.to_datetime(MP.opened_at.astype(str).str[:19]) - pd.Timestamp(0)).dt.total_seconds().mul(1000).astype("int64")
MAPM = {"LONG_HEAT_BLOCK": ["LONG_HEAT_BLOCK"], "LONG_MEGACAP_BLOCK": ["LONG_MEGACAP_BLOCK"], "LONG_CHOP_BURST": ["LONG_CHOP_BURST"],
        "PAIR_RSI_MOMENTUM_LOADX": ["LONG_RSI_MOM_LOADX"], "CALM3D_DMI": ["CALM3D_DMI_DI", "CALM3D_DMI_ADX"],
        "CALM3D_BTC_ATR_MIN": ["CALM3D_BTC_ATR_MIN"], "CALM3D_REENTRY": ["CALM3D_REENTRY"]}
SG = json.load(open("reports/SCOUT_REVERT_GATES.json")).get("gates", {})
MAPF = {"LONG_HEAT_BLOCK": "HEAT", "LONG_MEGACAP_BLOCK": "MEGACAP", "LONG_CHOP_BURST": "CHOP_BURST", "PAIR_RSI_MOMENTUM_LOADX": "LOADX"}


def real_cohorts(gate):
    out = {}
    m = MP[MP.stack_block_reason.isin(MAPM.get(gate, []))]
    if len(m):
        out["master (as traded)"] = pd.DataFrame(dict(t=m.t.values, pct=pd.to_numeric(m.pnl_percentage, errors="coerce").values))
    it = (SG.get(MAPF.get(gate, ""), {}) or {}).get("items", {}) or {}
    f = [(int(v["t"]), float(v["sim1"])) for v in it.values() if v.get("sim1") is not None]
    if f:
        out["forward (scout sim1)"] = pd.DataFrame(f, columns=["t", "pct"])
    return out


def state_read(df, split):
    R = FM.raw_tags(df.t.values); lab, la, lb = FM.splits(R)[split]
    res = {}
    for st, nm in (("A", la), ("B", lb)):
        m = lab == st
        x = df.pct.values[m]; dd = pd.Series(df.t.values[m] // DAY).nunique()
        res[st] = f"{len(x)} · {dd} d · {np.mean(x):+.3f}" if len(x) else "–"
        res[st + "v"] = np.mean(x) if len(x) else np.nan
    return res


chk = pd.concat([surv, gs[gs.gate.isin(MAPM)], MX[(MX.gate == "LONG_HEAT_BLOCK")]]).drop_duplicates(["gate", "split"])
L += ["## R. Real-fill cross-check (refute-only) — master stack blocks (as traded) and the scout's forward re-priced refusals, per state", "",
      "Flag = the real cohort's state ordering points the other way (yr5 says the filter is worth more in state X; real fills say the opposite).", "",
      "| gate | split | yr5 A / B avg | source | A: N · days · avg | B: N · days · avg | flag |", "|---|---|---|---|---|---|---|"]
for _, r in chk.iterrows():
    rc = real_cohorts(r.gate)
    if not rc:
        L.append(f"| {r.gate} | {r.split} | {r.avgA:+.3f} / {r.avgB:+.3f} | no real-fill cohort (gate never let a real fill through / no scout tracker) | – | – | n/a |")
        continue
    for src, df in rc.items():
        s = state_read(df, r.split)
        flag = ""
        if np.isfinite(s["Av"]) and np.isfinite(s["Bv"]):
            flag = "AGREES" if np.sign(s["Av"] - s["Bv"]) == np.sign(r.avgA - r.avgB) else "⚠ OPPOSITE"
        else:
            flag = "one state empty"
        L.append(f"| {r.gate} | {r.split} | {r.avgA:+.3f} / {r.avgB:+.3f} | {src} | {s['A']} | {s['B']} | {flag} |")
L.append("")
open(f"{S}/frm_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
