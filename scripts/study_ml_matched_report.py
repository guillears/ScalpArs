#!/usr/bin/env python3
"""ML MATCHED-LONG SHADOW study (2026-10-26) — step 4: verdicts + the markdown report.

Verdict rule (pre-stated in the brief; master over backtest, backtest refute-only):
  · SMALL-SIZE LIVE PROBE candidate — the LIVE cell clears the whole allow bar (study_ml_matched_analyse.verdict) AND the replay cell does
    not refute it (replay avg ≥ 0 or replay Δ vs its same-days control ≥ 0).
  · DECIDED no — LIVE avg ≤ 0, or LIVE avg below its same-days unmatched control, or LIVE N < 10 (no evidence to move a locked block),
    or LIVE positive but the replay refutes (replay avg < 0 AND replay Δ vs control < +0.05).
  · OPERATOR CALL — everything else (LIVE positive and ≥ its control, fails only N / days / confidence / concentration, replay not refuting).
    venv/bin/python -W ignore scripts/study_ml_matched_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_matched_common as M  # noqa: E402

PFX = M.PFX
OUT = PFX + ".md"


def verdict(lv, rp):
    if lv is None or lv.N < 10:
        return "DECIDED no", "LIVE N<10 — no evidence to move the block"
    if lv.avg <= 0:
        return "DECIDED no", f"LIVE avg {lv.avg:+.3f} ≤ 0"
    if not (lv.delta_vs_ctl >= 0):
        return "DECIDED no", f"LIVE below same-days control (Δ {lv.delta_vs_ctl:+.3f})"
    refute = rp is not None and rp.avg < 0 and not (rp.delta_vs_ctl >= 0.05)
    if lv.pass_bar and not refute:
        return "SMALL-SIZE PROBE", "clears the bar"
    if refute:
        return "DECIDED no", f"LIVE +{lv.avg:.3f} fails [{lv.fails}]; replay refutes ({rp.avg:+.3f}, Δ {rp.delta_vs_ctl:+.3f})"
    return "OPERATOR CALL", f"LIVE fails only [{lv.fails}]"


def fmt(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def main():
    T = pd.read_csv(PFX + "_cells.csv")
    E = pd.read_csv(PFX + "_episodes.csv", low_memory=False)
    S = pd.read_csv(PFX + "_shadows.csv.gz", low_memory=False)
    C = pd.read_csv(PFX + "_controls.csv.gz", low_memory=False)
    P = pd.read_csv(PFX + "_parity.csv")
    LV = T[T.dataset == "LIVE"].set_index("cell")
    RPd = T[T.dataset == "RPL"].set_index("cell")
    cells = list(dict.fromkeys(list(LV.index) + list(RPd.index)))
    rows = []
    for c in cells:
        lv = LV.loc[c] if c in LV.index else None
        rp = RPd.loc[c] if c in RPd.index else None
        v, why = verdict(lv, rp)
        kind = (lv if lv is not None else rp).kind
        rows.append(dict(cell=c, kind=kind, verdict=v, why=why,
                         **{f"live_{k}": (lv[k] if lv is not None else np.nan) for k in ("N", "days", "WR", "avg", "LB_bonf", "ctl_avg", "delta_vs_ctl", "top_day_share", "top_pair_share", "fails")},
                         **{f"rpl_{k}": (rp[k] if rp is not None else np.nan) for k in ("N_per_seed", "days", "WR", "avg", "LB_bonf", "ctl_avg", "delta_vs_ctl")}))
    V = pd.DataFrame(rows)
    V.to_csv(PFX + "_verdicts.csv", index=False)

    L = []
    W = L.append
    ep = E
    live = ep[ep.cohort.isin(["PJUL", "JSEP", "CWOCT"])]
    rpl = ep[ep.cohort == "RPL"]
    CL = C[(C.cohort == "CTL_LIVE") & C.pct_taker.notna()]
    CR = C[(C.cohort == "CTL_RPL") & C.pct_taker.notna()]
    MR = C[C.cohort == "MREAL"]

    W("# ML MATCHED-LONG SHADOW — should refused C/W-matched momentum longs be allowed? (2026-10-26)\n")
    W("Study only: no engine, config, UI, DECISION_LOG or CURRENT_STATE change. Scripts: `scripts/study_ml_matched_{extract,fetch,build,parity,analyse,report}.py` "
      "(+ `study_ml_matched_common.py`). All P&L is % per fill at 1×, priced by the validated ML exit replica at **today's** exit (RP.BASE), taker entry fee for shadows and controls alike.\n")

    W("## 0. Logs check — are matched-long refusals still recorded, with enough to price them?\n")
    W("**Short answer: only partly.** Nothing records the specific pattern labels of a refusal today. The live engine still logs every refusal, but each surface is missing something:\n")
    W("| source | from → to | per refusal | pattern info | indicator state | usable? |")
    W("|---|---|---|---|---|---|")
    W("| (a) orders export `scalpars_orders_*` | – | fills only; a refused long never becomes an Order | – | – | no |")
    W("| (a) decisions export `/api/decisions/export.csv` (`reports/BASELINE22-24_*_LIVE_decisions.csv`) | last 1–7 days per download | BLOCK rows **aggregated per 5 min × pair × gate** (t bucket, pair, count) — no price, no seconds | none | none | count only |")
    W("| (a) phantoms export `scalpars_phantoms_*` | **2026-07-06 → 07-26** (seeding retired 07-26/07-30, `_seed_phantom_flip` is a no-op) | pair, second, entry price, full entry_* stamps, July-exit outcome | **yes — `entry_cohort` (e.g. `W2+W6`)** | yes | **yes (303 rows → 209 episodes)** |")
    W("| (b) `_record_filter_block('LONG_UNMATCHED_ONLY')` | live | in-memory counter + `BotState.filter_block_counts_json` (counts only; reset on paper reset) | none | none | count only |")
    W("| (b) decision journal `/opt/scalpars-data/journal/decisions-*.jsonl` (server, 45-day retention) | server ≈ 09-11 → now; **local copy only 09-20 → 09-24** (`replay/year/journal/calib_live`) | BLOCK gate=LONG_UNMATCHED_ONLY: pair, ms time, room flag, **pair indicator snapshot (ctx: price, RSI, ADX, EMAs, ATR, range, volume)**; the SCAN header has BTC RSI / ADX / ADX-prev / slope / off30d / bull | **none** (rebuildable: engine pattern functions on ctx + SCAN + rebuilt BTC RSI-prev / ATR / EMA13-50 gap) | yes | yes (88 rows → 33 episodes); the server copy 09-25 → 10-26 needs the EB log bundle, broken since 10-19 |")
    W("| (c) CloudWatch `web.stdout.log` (14-day retention) | **2026-10-12 16:29 → 10-26 05:57** | `[LONG_UNMATCHED_ONLY] PAIR: LONG blocked — matched a pattern (c_any=…, w_any=…)` + `[OPEN_FILTERED] … LONG <conf>`: pair, ms time, confidence | **family only (C / W / C+W)** | none (rebuilt from klines; price = last closed 1m close) | yes (117 rows → 76 episodes) |")
    W("| (d) `reports/` July phantom CSVs | 07-06 → 07-26 | as (a) phantoms | yes | yes | same rows as (a) |")
    W("| engine replay journals yr6 / yr6a (today's config at 01419d4) | 2025-01-01 → 2026-10-11, 3 seeds | same BLOCK lines as the live journal | rebuilt (99.8 % of refusals rebuild as matched) | yes | backtest, refute-only (32,919 rows → 15,789 episodes) |\n")
    W("AWS access worked (read-only `aws logs filter-log-events`); nothing was written.\n")
    W("**Scout-line proposal (observe-only, NOT built):** a `LONG_UNMATCHED_ONLY` refusal recorder. The block already computes everything at the refusal site (`services/trading_engine.py` ~L12889–12950: `_um_pats` / `_um_cohort` and the full `_um_ef` entry-field dict are built and then handed to the retired `_seed_phantom_flip` no-op). "
      "It needs: ① one journal line per refusal: `_djournal.note('MATCHED_LONG', pair, price=current_price, conf, labels=_um_pats, ef=_um_ef, room)` (file-only, no DB, no trading-path effect) — or a `matched_long_refusals` table if it must ride the CSV export (D12: real model columns); "
      "② the decisions export carrying those rows un-aggregated (pair, second, price, labels, entry_* stamps); ③ a scout script pricing each row with the ML replica at today's exit, one per pair at a time, with the frozen cells below; "
      "④ D11/D12 if it gets a UI table (both text exports + CSV columns). Cost: ~650 bytes per refusal (≈ 10–25 / day live).\n")

    W("## 1. Plain-language summary\n")
    a = live; b = CL[CL.o_ms.isin(CL.o_ms)]
    W(f"- **Every refused matched long we can recover, priced at today's exit, loses money.** Live records (July phantoms, Sep journal, Oct CloudWatch): "
      f"{len(live)} shadow trades on {live.day.nunique()} days · WR {(live.y > 0).mean() * 100:.1f} % · **{live.y.mean():+.3f} %/fill**, versus the unmatched longs the bot actually took on the same days "
      f"({fmt(LV.loc['ALL matched'].ctl_avg)} %/fill, N {int(LV.loc['ALL matched'].ctl_N)}). The block is still doing its job: the matched longs are ~{abs(LV.loc['ALL matched'].delta_vs_ctl):.2f} pp/fill worse.")
    W(f"- It holds in **each live period separately** (July {fmt(T[(T.dataset == 'PJUL') & (T.cell == 'ALL matched')].avg.iloc[0])}, Sep {fmt(T[(T.dataset == 'JSEP') & (T.cell == 'ALL matched')].avg.iloc[0])} on only 33 trades / 5 days, Oct {fmt(T[(T.dataset == 'CWOCT') & (T.cell == 'ALL matched')].avg.iloc[0])}), for C, W and C+W alike, and in every BTC regime / 1h-slope / 24h / 30d-high / breadth split.")
    W(f"- The **backtest** (21 months, 3 seeds, {len(rpl):,} shadow trades) also has matched longs losing ({rpl.y.mean():+.3f} %/fill, WR {(rpl.y > 0).mean() * 100:.1f} %) — but there the unmatched longs lose about as much "
      f"({CR.pct_taker.mean():+.3f}), so the backtest says the block neither helps nor hurts; it cannot support allowing anything (its absolute numbers are negative everywhere).")
    W("- **Least-bad cell: W2 without any C pattern** (BTC RSI 50–65 tailwind). Live 82 · 58.5 % · −0.05 %/fill, still below its control; it is the same cell whose July re-enable reverted (22 · 45 % · −$250). Its positive sub-slices (STRONG_BULL, near the 30-day high, BTC 1h rising) are 32–53 trades with one or two days carrying more than all of the profit.")
    W("- **No cell clears the bar; none is a probe candidate.** Recommendation: keep `long_unmatched_only = true`. If the operator wants the question kept alive, build the observe-only refusal recorder above (no trading change) and read two frozen cells at N ≥ 30 / 10 days.\n")

    W("## 2. Data, parity and pricing checks\n")
    g = S.groupby("cohort")
    W("| cohort | refusals | blocked by today's post-block gates | priced | one-per-pair episodes | tick / 1m paths |")
    W("|---|---|---|---|---|---|")
    for co in ["PJUL", "JSEP", "CWOCT", "RPL"]:
        x = S[S.cohort == co]
        gg = x.gate.fillna("").value_counts()
        gt = ", ".join(f"{k} {v}" for k, v in gg.items() if k) or "0"
        W(f"| {co} | {len(x):,} | {gt} | {x.pct.notna().sum():,} | {int(x.episode.sum()):,} | {(x.src == 'tick').sum():,} / {(x.src == '1m').sum():,} |")
    pj = S[S.cohort == "PJUL"]
    cw = S[S.cohort == "CWOCT"]
    fam_ok = (cw.family.replace({"NONE": ""}) == cw.family_log.fillna("")).mean() * 100
    R_ = P[P.rebuilt]
    famf = lambda s: ("C" if any(t.startswith("C") for t in str(s).split("+") if t and s == s) else "") + ("W" if any(t.startswith("W") for t in str(s).split("+") if t and s == s) else "")
    pf = (R_.stamp_label.map(famf) == R_.rebuilt_label.map(famf)).mean() * 100
    um = R_[R_.stamp_label.isna()]
    W("")
    W(f"- **Cohort = engine events, not a proxy** (FATAL rule): every shadow is a refusal the engine itself logged / journalled / seeded (live) or journalled (replay) — the same bars by construction. Labels are the engine's own `_compute_pattern_c/w_match`.")
    W(f"- Label rebuild parity: replay refusals rebuilt as matched **99.8 %** (journal ctx + SCAN BTC RSI/ADX exact; BTC RSI-prev / ATR / EMA13-50 gap rebuilt — ADX median |Δ| 0, RSI-prev median |Δ| 0.03 vs replay fill stamps); "
      f"Sep journal 100 %; CloudWatch family (C / W / C+W) rebuilt = logged on **{fam_ok:.1f} %** (specific labels used only on those rows, the logged family elsewhere).")
    W(f"- Kline-only rebuild on the **real master fills** since 09-20 (the CloudWatch path): family parity {pf:.1f} %; {int((um.rebuilt_label.notna()).sum())} of {len(um)} unmatched fills rebuild as matched "
      f"(BTC RSI from the forming bar, median |Δ| 0.6) → specific labels from klines are approximate; the logged family is authoritative.")
    W(f"- July phantoms keep the **live-recorded** label: re-labelling them with today's thresholds agrees on {(pj.label_live == pj.label.fillna('')).mean() * 100:.0f} % — the gap is the July stamps lacking `entry_btc_trend_gap_pct` (W2 / C2 need it), not a threshold change.")
    lc = C[C.cohort == "CTL_LIVE"]
    W(f"- Replica on the {len(lc)} master KEEP control fills (own fee): {fmt(lc.pct_realfee.mean())} %/fill vs master stack {fmt(lc.y.mean())}; base exit reason 88 % equal (mismatches = BASE/B1 fills that traded older exit stacks; DL 380's 100 % is the as-was replica). "
      f"PJUL episodes: July phantom exit {pj[pj.episode & pj.july_pct.notna()].july_pct.mean():+.3f} vs today's replica {pj[pj.episode & pj.july_pct.notna()].pct.mean():+.3f} — today's exit does **not** rescue them.")
    W(f"- Probe routing: shadows never traded, so none is probe-routed; controls and MREAL exclude every `is_probe` fill (W2 lesson). Pattern labels are kept separate from routing.\n")

    def tbl(kind, filt=None, title=""):
        x = V[V.kind == kind]
        if filt is not None:
            x = x[filt(x)]
        W(title)
        W("| cell | LIVE N | days | WR % | avg %/fill | LB (Bonf.) | same-days ctl | Δ vs ctl | top day / pair share | RPL N/seed | RPL days | RPL WR | RPL avg | RPL Δ vs ctl | verdict |")
        W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in x.itertuples():
            n = "–" if np.isnan(r.live_N) else int(r.live_N)
            d = "–" if np.isnan(r.live_days) else int(r.live_days)
            sh = "–" if np.isnan(r.live_top_day_share) else f"{r.live_top_day_share:.2f} / {r.live_top_pair_share:.2f}"
            W(f"| {r.cell} | {n} | {d} | {fmt(r.live_WR, 1, False)} | {fmt(r.live_avg)} | {fmt(r.live_LB_bonf)} | {fmt(r.live_ctl_avg)} | {fmt(r.live_delta_vs_ctl)} | {sh} | "
              f"{fmt(r.rpl_N_per_seed, 0, False)} | {fmt(r.rpl_days, 0, False)} | {fmt(r.rpl_WR, 1, False)} | {fmt(r.rpl_avg)} | {fmt(r.rpl_delta_vs_ctl)} | **{r.verdict}** — {r.why} |")
        W("")

    W("## 3. Per-cell tables\n")
    W(f"Bar for an *allow* cell: N ≥ 30 on ≥ 10 days · WR ≥ {M.BE_WR} % · avg ≥ +0.10 %/fill · day-block bootstrap lower bound > 0 at the Bonferroni level over all cells tested "
      f"(LIVE m = {int(T[T.dataset == 'LIVE'].m_tests.iloc[0])}, RPL m = {int(T[T.dataset == 'RPL'].m_tests.iloc[0])}) · ≥ the same-days unmatched control · no day / pair ≥ 50 % of the net · then a 30–50 % haircut. "
      "LIVE = PJUL + JSEP + CWOCT (decisive). RPL = replay (refute-only; N per seed, CI clustered by day across the 3 seeds). Δ vs ctl = cell avg − the unmatched control's avg on the same days (same replica, same fee). "
      "'top day / pair share' is shown only when the net is positive. Regime cells are market-wide: read the **days** column.\n")
    tbl("pattern", title="### 3a. By pattern (single labels, families, C-free W, exact combos ≥ 30)\n")
    tbl("pattern×regime", filt=lambda x: (x.live_N >= 10) | (x.rpl_N_per_seed >= 30),
        title="### 3b. Pattern family × regime (BTC regime label · BTC 1h EMA20 slope sign · BTC 24h return sign · distance from 30-day high · breadth = share of top-50 above 1h EMA50; definitions = the ML_REGIME_WL study's `study_ml_regime_wl_build.build`)\n")
    tbl("era", title="### 3c. By era (LIVE is all 2026-H2; per live period see 3d)\n")
    W("### 3d. Live periods separately\n")
    W("| period | cell | N | days | WR % | avg %/fill | same-days ctl N | ctl avg | Δ |")
    W("|---|---|---|---|---|---|---|---|---|")
    for co, nm in [("PJUL", "Jul 06–26 (phantoms)"), ("JSEP", "Sep 20–24 (journal)"), ("CWOCT", "Oct 12–26 (CloudWatch)")]:
        for c in ["ALL matched", "C-only (no W)", "W-only (C-free)", "C+W", "W2 C-free"]:
            x = T[(T.dataset == co) & (T.cell == c)]
            if len(x):
                r = x.iloc[0]
                W(f"| {nm} | {c} | {int(r.N)} | {int(r.days)} | {r.WR:.1f} | {fmt(r.avg)} | {int(r.ctl_N)} | {fmt(r.ctl_avg)} | {fmt(r.delta_vs_ctl)} |")
    W("")
    W("### 3e. Matched momentum longs that really traded (MREAL — doors / bypasses, non-probe, master)\n")
    W(f"{len(MR)} fills · as traded {MR.y.mean():+.3f} %/fill · today's replica {MR.pct_taker.mean():+.3f} · WR {(MR.pct_taker > 0).mean() * 100:.0f} % — "
      + ", ".join(f"{k} {len(g)}·{g.pct_taker.mean():+.2f}" for k, g in MR.groupby("label") if len(g) >= 3) + ". Same direction as the shadows (descriptive; door-selected).\n")

    W("## 4. Verdicts\n")
    vc = V.verdict.value_counts()
    W(f"{len(V)} cells: " + ", ".join(f"**{k}** {v}" for k, v in vc.items()) + ". Full list with reasons: `reports/ML_MATCHED_SHADOW_2026-10-26_verdicts.csv`.\n")
    W("- **SMALL-SIZE LIVE PROBE: none.** No cell clears the bar on live data; the replay refutes every positive live slice. (Had one cleared: 0.05 lev = 1× at 0.2 invest, revert on N ≥ 10 with WR < 55 % or Σ < 0, read at N 30.)")
    W("- **DECIDED no:** every family (C-only, C-free W, C+W, all matched) and every single label — negative live, below the same-days control, in all three live periods.")
    oc = V[V.verdict == "OPERATOR CALL"]
    if len(oc):
        W("- **OPERATOR CALL:** " + "; ".join(f"{r.cell} (LIVE {int(r.live_N)} · {r.live_WR:.0f} % · {r.live_avg:+.3f}; {r.why})" for r in oc.itertuples()) + ".")
    else:
        W("- **OPERATOR CALL:** none.")
    W("- **Frozen observe candidates (if the recorder is built; pre-registered now, never re-fit):** ① `W2 C-free` (any regime) · ② `W2 C-free × BTC regime STRONG_BULL`. "
      "Gate per cell, forward fills only: N ≥ 30 on ≥ 10 days · WR ≥ 61.5 % · avg ≥ +0.10 %/fill · day-bootstrap LB > 0 at α = 0.025 (Bonferroni over 2) · ≥ same-days unmatched control · no day / pair ≥ 50 %. "
      "Pass → OPERATOR CALL on a 1× probe; fail → DECIDED no, recorder off.\n")

    W("## 5. Could NOT test\n")
    W("- **Server journal 09-25 → 10-26** (ctx snapshots for every refusal) — the EB log bundle has been broken since 10-19; CloudWatch carries only the C/W family, so October specific labels are kline-rebuilt (approximate).")
    W("- **Aug 1 → Sep 19 live refusals** — no phantom seeding (retired), journal not yet shipped (Sep-18) / not local, CloudWatch retention 14 days. A 7-week hole.")
    W("- **Slot competition / portfolio effects** — a matched long would take one of the 4 slots and can displace an unmatched fill; shadows ignore slots (the replay journal only lists refusals with room = true).")
    W("- **LONG_CHOP_BURST** (needs the live fill book) not applied to shadows; **LONG_HEAT_BLOCK** fails open on CloudWatch rows (no breadth reading) and the mega-cap gate uses the daily top-50 rank as a proxy outside the July stamps.")
    W("- **Entry order type / slippage** — every shadow priced at a taker entry at the logged / journalled price (CloudWatch: last closed 1m close, ≤ 60 s stale); no queue or maker model.")
    W("- **Sizing cells** — matched cells would trade at their pattern-cell size (all LONG cells 1× today); % is leverage-invariant, so $ effects are not modelled.")
    W("- **July phantoms ran under the July ENTRY stack**: gates removed since then (e.g. FAKE_BULL_GUARD) could have refused rows that today's stack would see; gates added since were re-applied (blacklist, heat, mega-cap, LOADX).")
    W("- **Funding / order-book state** at refusal time — not recorded on any surface.")
    W("- **Backtest absolute level** — the replay ML sleeve itself loses (unmatched control −0.16 %/fill vs +0.11 live), so the replay is read only relative to its own control and only to refute.\n")

    W("## Files\n")
    for f in ["_cloudwatch_refusals.csv", "_replay_blocks.csv.gz", "_shadows.csv.gz", "_controls.csv.gz", "_episodes.csv", "_regime.csv", "_parity.csv", "_cells.csv", "_verdicts.csv"]:
        W(f"- `reports/ML_MATCHED_SHADOW_2026-10-26{f}`")
    W("- price-path top-up: `reports/ML_MATCHED_SHADOW_2026-10-26_fetch/` (public fapi klines, 300 calls, ≤ 400 weight/min)")
    open(OUT, "w").write("\n".join(L) + "\n")
    print("→", OUT, "| verdicts", vc.to_dict())


if __name__ == "__main__":
    main()
