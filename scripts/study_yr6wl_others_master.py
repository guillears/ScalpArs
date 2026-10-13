#!/usr/bin/env python3
"""YR6 winners-vs-losers, AREA = OTHERS — replay vs REAL fills (master pool, today's stack) for the sleeves that have real fills:
FLIP-short (FAN flips), Spike-Fade, BullRun-Long, BearRun-Short (SURGE_LONG has no master fill).

  venv/bin/python scripts/study_yr6wl_others_master.py

For each sleeve:
  ① master kept (stack_keep, non-probe) N · WR · avg stack_pct  vs  replay (3 seeds) on the SAME active days (days the live bot
     took ANY fill — bot offline / sleeve-disabled days excluded from the replay side)
  ② match replay ↔ master by pair and |open time| ≤ 20 min (nearest; each seed separately) → matched outcome Δ (exit-side gap),
     entry-price Δ, exit-reason crosstab, live trough vs replay trough (did live ever reach the replay's stop depth?)
  ③ live-only and replay-only fills (selection-side gap)
  ④ decomposition: master avg − replay avg = matched Δ share + selection share (two ways: per-fill means and Σ)
Output: reports/YR6_WL_OTHERS_master.txt + reports/YR6_WL_OTHERS_master_matches.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_others_common import REP, load, day_ci

D = load()
M = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
M["t"] = pd.to_datetime(M.opened_at, format="ISO8601"); M["day"] = M.t.dt.floor("D")
active_days = set(M.day)
MAP = {"FLIP-short": "FLIP:FAN_RATIO_GATE", "Spike-Fade": "SPIKE_FADE", "BullRun-Long": "BULLRUN_LONG", "BearRun-Short": "BEARRUN_SHORT",
       "SURGE_LONG": "SURGE_LONG"}
OUT = []; MATCH = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


def line(y, ns=1, days=None):
    y = pd.Series(y).dropna()
    if not len(y):
        return "N 0"
    return f"N {len(y) / ns:5.1f} · WR {(y > 0).mean() * 100:4.0f}% · avg {y.mean():+.3f} · Σ {y.sum() / ns:+.2f}" + (f" · {days} d" if days else "")


for SL, ST in MAP.items():
    m = M[(M.entry_strategy == ST) & (M.is_probe != True)].copy()
    P("=" * 130); P(f"{SL}  (master entry_strategy {ST})")
    if not len(m):
        P("  no master fills — sleeve has no real-fill check (state it)"); continue
    mk = m[m.stack_keep == True].copy(); mk["y"] = pd.to_numeric(mk.stack_pct, errors="coerce")
    m["y_raw"] = pd.to_numeric(m.pnl_percentage, errors="coerce")
    lo, hi = m.day.min(), m.day.max()
    P(f"  master window {lo.date()} → {hi.date()} · all {len(m)} ({line(m.y_raw)} as traded) · kept today's stack {line(mk.y, days=mk.day.nunique())}")
    blocked = m[m.stack_keep != True]
    P(f"  master blocked by today's stack: {len(blocked)} · {line(blocked.y_raw)} · reasons {blocked.stack_block_reason.value_counts().to_dict()}")
    r = D[(D.sleeve == SL) & (D.day >= lo) & (D.day <= hi)].copy()
    ra = r[r.day.isin(active_days)]
    if SL == "FLIP-short":  # the live flip entry path was dead 2026-07-08 → 2026-08-11 (TypeError, memory feedback_absence_is_a_logs_question)
        ra = ra[~((ra.t >= "2026-07-08") & (ra.t < "2026-08-12"))]
        P("  (flip-dead window 2026-07-08 → 08-11 removed from the replay side)")
    P(f"  replay same window: {line(r.y, 3, r.day.nunique())} · on live-active days only: {line(ra.y, 3, ra.day.nunique())}")
    for rg, h in ra.groupby("regime"):
        mh = mk[mk.t.between(h.t.min() - pd.Timedelta(days=1), h.t.max() + pd.Timedelta(days=1))]
        P(f"     {rg}: replay {line(h.y, 3)} · master kept {line(mh.y)}")
    # sub-windows: washed-out window Jun-18→Jul-2
    for lab, a, b in (("washed-out Jun-17→Jul-2", "2026-06-17", "2026-07-03"), ("after Jul-2", "2026-07-03", "2026-12-31")):
        mm = mk[(mk.t >= a) & (mk.t < b)]; rr = ra[(ra.t >= a) & (ra.t < b)]
        P(f"     {lab:26}: master kept {line(mm.y, days=mm.day.nunique())} | replay {line(rr.y, 3, rr.day.nunique())}")
    # ② matching
    rows = []
    for sd in sorted(ra.seed.unique()):
        rs = ra[ra.seed == sd]
        used = set()
        for i, x in m.iterrows():
            c = rs[(rs.pair == x.pair)]
            if not len(c):
                continue
            dt = (c.t - x.t).abs()
            j = dt.idxmin()
            if dt[j] <= pd.Timedelta(minutes=20) and j not in used:
                used.add(j); q = rs.loc[j]
                ep_m, ep_r = float(x.entry_price), float(q.entry_price)
                rows.append(dict(sleeve=SL, seed=sd, pair=x.pair, t_live=x.t, t_replay=q.t, dt_s=(q.t - x.t).total_seconds(), kept=bool(x.stack_keep),
                                 live_pct=x.y_raw, live_stack=pd.to_numeric(x.stack_pct, errors="coerce"), replay_pct=q.y,
                                 entry_px_diff_pct=(ep_r / ep_m - 1) * 100 * (-1 if x.direction == "SHORT" else 1),
                                 live_reason=str(x.close_reason).split(" ")[0], replay_reason=str(q.close_reason).split(" ")[0],
                                 live_trough=pd.to_numeric(x.trough_pnl, errors="coerce"), replay_trough=pd.to_numeric(q.trough_pnl, errors="coerce"),
                                 live_peak=pd.to_numeric(x.peak_pnl, errors="coerce"), replay_peak=pd.to_numeric(q.peak_pnl, errors="coerce"),
                                 live_hold_min=(pd.to_datetime(x.closed_at, format="ISO8601") - x.t).total_seconds() / 60,
                                 replay_hold_min=(pd.to_datetime(q.closed_at, format="ISO8601") - pd.to_datetime(q.opened_at, format="ISO8601")).total_seconds() / 60))
        r.loc[r.seed == sd, "matched"] = r.loc[r.seed == sd].index.isin(used)
    X = pd.DataFrame(rows)
    if not len(X):
        P("  no replay ↔ master matches"); continue
    MATCH.append(X)
    Xk = X[X.kept]
    n_live_kept_matched = Xk.groupby(["pair", "t_live"]).ngroups
    P(f"  ② matched (pair, ≤ 20 min): {len(X)} seed-matches · {X.groupby(['pair', 't_live']).ngroups} distinct live fills of {len(m)} "
      f"({n_live_kept_matched} of {len(mk)} kept)")
    P(f"     kept & matched: live {line(Xk.live_pct)}  vs  replay same entries {line(Xk.replay_pct)} → matched Δ (live − replay) "
      f"{(Xk.live_pct - Xk.replay_pct).mean():+.3f}/fill (median {(Xk.live_pct - Xk.replay_pct).median():+.3f}); "
      f"entry-px Δ (replay worse +) median {Xk.entry_px_diff_pct.median():+.3f}%, |dt| median {Xk.dt_s.abs().median():.0f}s")
    P("     exit-reason crosstab (live rows × replay cols), kept & matched:\n" + pd.crosstab(Xk.live_reason, Xk.replay_reason).to_string())
    stop_r = Xk[Xk.replay_reason.str.contains("STOP") & ~Xk.live_reason.str.contains("STOP")]
    if len(stop_r):
        P(f"     replay STOPPED but live did not: {len(stop_r)} seed-matches · live {line(stop_r.live_pct)} · replay {line(stop_r.replay_pct)} · "
          f"live trough median {stop_r.live_trough.median():+.2f} vs replay trough median {stop_r.replay_trough.median():+.2f}; "
          f"replay hold median {stop_r.replay_hold_min.median():.1f} min vs live {stop_r.live_hold_min.median():.1f}")
    stop_l = Xk[~Xk.replay_reason.str.contains("STOP") & Xk.live_reason.str.contains("STOP")]
    if len(stop_l):
        P(f"     live STOPPED but replay did not: {len(stop_l)} · live {line(stop_l.live_pct)} · replay {line(stop_l.replay_pct)}")
    # distinct-fill decomposition (seed copies averaged per live fill)
    Q = Xk.groupby(["pair", "t_live"]).agg(live=("live_pct", "first"), rep=("replay_pct", "mean"), lr=("live_reason", "first"),
                                           rstop=("replay_reason", lambda s: s.str.contains("STOP_LOSS").mean())).reset_index()
    Q["lstop"] = Q.lr.str.contains("STOP_LOSS"); Q["d"] = Q.live - Q.rep
    for lab, mm in (("both trail/TP", ~Q.lstop & (Q.rstop == 0)), ("replay stop, live not (any seed)", ~Q.lstop & (Q.rstop > 0)),
                    ("live stop, replay not", Q.lstop & (Q.rstop < 1)), ("both stop", Q.lstop & (Q.rstop == 1))):
        if mm.sum():
            P(f"     distinct live fills · {lab:34}: {mm.sum():3d} · live {Q.live[mm].mean():+.3f} vs replay {Q.rep[mm].mean():+.3f} → Δ {Q.d[mm].mean():+.3f} · Σ Δ {Q.d[mm].sum():+.2f}")
    for lab, a, b in (("≤ Jul-2", "2000-01-01", "2026-07-03"), ("> Jul-2", "2026-07-03", "2100-01-01")):
        qq = Q[(Q.t_live >= a) & (Q.t_live < b)]
        if len(qq):
            P(f"     distinct matched kept fills {lab}: {len(qq)} · live {qq.live.mean():+.3f} vs replay {qq.rep.mean():+.3f} → Δ {qq.d.mean():+.3f} (median {qq.d.median():+.3f})")
    # ③ selection
    lm = set(zip(X.pair, X.t_live))
    live_only = mk[[(p, t) not in lm for p, t in zip(mk.pair, mk.t)]]
    P(f"  ③ live kept but NO replay fill on any seed: {line(live_only.y)} ({', '.join(live_only.pair.str.replace('USDT', '') + ' ' + live_only.t.dt.strftime('%m-%d'))})")
    ro = ra[ra.index.isin(r[r.matched != True].index)]
    P(f"     replay fills with no live fill (active days): {line(ro.y, 3, ro.day.nunique())} · replay matched: {line(ra[ra.index.isin(r[r.matched == True].index)].y, 3)}")
    # ④ decomposition (two ways)
    A_live = mk.y.mean(); A_rep = ra.y.mean()
    A_live_m = Xk.drop_duplicates(["pair", "t_live"]).live_stack.mean(); A_rep_m = Xk.replay_pct.mean()
    P(f"  ④ gap master-kept {A_live:+.3f} − replay-active {A_rep:+.3f} = {A_live - A_rep:+.3f}/fill · of which exit-side on the SAME entries "
      f"(live {A_live_m:+.3f} − replay {A_rep_m:+.3f}) = {A_live_m - A_rep_m:+.3f}; the rest is selection (which fills each side took)")
    P(f"     cross-check Σ: live kept Σ {mk.y.sum():+.2f} over {len(mk)} vs replay Σ/seed {ra.y.sum() / 3:+.2f} over {len(ra) / 3:.1f}/seed")

if MATCH:
    pd.concat(MATCH).to_csv(os.path.join(REP, "YR6_WL_OTHERS_master_matches.csv"), index=False)
open(os.path.join(REP, "YR6_WL_OTHERS_master.txt"), "w").write("\n".join(OUT) + "\n")
