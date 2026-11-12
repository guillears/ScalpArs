#!/usr/bin/env python3
"""👁 Scout — ML_FIXED_TP watch line (registered 2026-11-05, DECISION_LOG 479; OBSERVE ONLY — never changes config, no bot API; public 1h klines only for the
ML_MKT_HOT Market-hot (% of the rolling top-100 up over 4 h) — 🌡 V2, DECISION_LOG 489, replaced the frozen 108 on 2026-11-06 — of fills that carry no engine stamp and lie past the 11-06 study cache, through the budgeted cache of scripts/scout_ml_mkt_hot_watch.py).

QUESTION   Would a plain fixed take-profit (+0.5 / +1 / +2 / +3 %) beat today's momentum-long exit (runner trail + ladder), on every full-size fill and on the
           fills the ML_MKT_HOT candidate filter keeps? Operator 11-05: "i find it very important … make sure in scout we are tracking this for momentum longs".
COHORT     (operator 11-05: "we have a huge history, the decision will be based on forward + history") — HISTORY + FORWARD, re-read in full every run:
           MASTER = reports/MASTER_POOL_stacked.csv full-size momentum longs kept on today's stack (before = stack_pct + the second chance, below) · B25 = every export fill opened between
           the B25 start (2026-10-23 18:00) and this line's registration (DL 479 commit + 10 min) · FORWARD = every fill opened after it. Exports = Downloads +
           exports_archive/, dedup (opened_at[:19], pair, direction, entry_strategy) — CLOSED beats OPEN, newest export wins; probes out
           (sweep_separators._is_probe, both flags); operator-closed out; not already in the master (a rebuilt master absorbs batches without double counting);
           a fill closed < 60 min before the newest export waits for its 45-min post-exit record.
🔂 RE-BASE  (DECISION_LOG 534, 2026-11-12 — the momentum-long exit changed with the second chance, DL 532): "today's exit" = the runner trail /
           ladder / recovery hold + the SECOND CHANCE. Fills opened before the DL 532 commit + 10 min read their booked % + the frozen per-fill
           second-chance delta (scripts/ml_sc_history.py; master 123 fills, 38 changed, Σ +11.77); a pre-ship fill the second chance could have
           changed but has no delta row is UNPRICED (shown); later fills are booked with it. The TP side is unchanged (it reads the path from entry).
RULE       per fill from its recorded columns — scripts/study_ml_fixed_tp_recorded.tp_outcome, the registration read, reused unchanged.
VERDICT    re-read every run on HISTORY + FORWARD together (never forward-only): per line (A all fills · B kept by ML_MKT_HOT) and level, Σ Δ ≤ 0 → NO EDGE ·
           EXIT CANDIDATE (operator decides at a batch review; never automatic) iff mean Δ > 0 ∧ day sign-flip luck p against the BEST of the 4 levels ≤ 0.10
           (4,000, seed 7) ∧ no single fill and no single day ≥ 50 % of Δ ∧ the same sign in MASTER and in B25 + FORWARD · else KEEP OBSERVING. A gain from
           fills still open at 45 min is holding longer, not a TP (shown per level). A live change also needs the 30–50 % in-sample haircut, a joint
           entry × exit check and a pre-committed revert.
Usage: venv/bin/python -I scripts/scout_ml_fixed_tp_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG                  # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK                # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_ml_mkt_hot_watch as MH        # noqa: E402  (the Market-hot measure — % of the rolling top-100 up over 4 h, DL 489 — budgeted + cached)
import study_ml_fixed_tp_recorded as RT    # noqa: E402  (cohort + rule of the registration read, reused unchanged)
import review_math as RM                   # noqa: E402
import ml_sc_history as SCH                # noqa: E402  (🔂 DL 534: "today's exit" includes the second chance on pre-ship fills)

DL = 479
TPS = RT.TPS
P_MAX, SHARE_MAX = 0.10, 0.50
N_MIN = 30
BOOT_N, BOOT_SEED = 4000, 7
SETTLE_MIN = 60
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
CANDIDATE = "EXIT CANDIDATE (operator decides at a batch review; never automatic)"
BLOCKS = ("MASTER", "B25", "FORWARD")


def load_exports(paths=None):
    """→ (export fills of every read export — dedup, CLOSED beats OPEN, newest export wins —, the newest export's download time or None)."""
    fr, newest = [], None
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if not {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            continue
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        d = d.copy()                                    # export frames arrive fragmented (hundreds of columns)
        d["_m"] = t.value
        fr.append(d)                 # newest EXPORT wins by its download stamp, not the copy's file time (second review 479)
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(), newest
    d = pd.concat(fr, ignore_index=True).copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction", "entry_strategy"], keep="last")
    return d.reset_index(drop=True), newest


def cohort(paths=None, since=None, master=None, sc_ship_ms="auto", sc_delta=None):
    """→ (fills MASTER + B25 + FORWARD priced under every TP, the newest export time, N waiting for their record, operator-closed fills).
    master given (tests) → its own keys; else the RAW master keys (every master momentum long, kept or not)."""
    M = master if master is not None else RT.load_master()[0]
    keys = RT.master_keys(M) if master is not None else RT.master_keys()
    E, newest = load_exports(paths)
    waiting, ox = 0, pd.DataFrame(columns=["opened_at", "pair"])
    if len(E):
        E = E[E.status.astype(str).str.upper().eq("CLOSED") & (E.entry_strategy.astype(str) == "MOMENTUM") & (E.direction.astype(str) == "LONG")]
        E, ox = _GK.split_operator_closed(E) if len(E) else (E, ox)
        E, _ = RT.batch_rows(E, keys, block="B25", era="B25")
        c = pd.to_datetime(E.closed_at.astype(str).str.replace(" ", "T").str[:19], format="ISO8601", errors="coerce")
        ok = (c.notna() & (c <= newest - pd.Timedelta(minutes=SETTLE_MIN))) if newest is not None else pd.Series(False, index=E.index)
        waiting = int((~ok).sum())
        E = E[ok].copy()
        if since is not None:
            E.loc[pd.to_datetime(E.opened_at, format="mixed") >= since, ["block", "era"]] = "FORWARD"
    F = RT.finish(pd.concat([M.reindex(columns=RT.COLS), E.reindex(columns=RT.COLS)], ignore_index=True))
    F = sc_rebase(F, "before", sc_ship_ms, sc_delta)
    return RT.price(F), newest, waiting, ox


def sc_rebase(F, col, sc_ship_ms="auto", sc_delta=None):
    """🔂 (DL 534) F[col] → today's exit incl. the second chance (ml_sc_history.rebase); counts in F.attrs["sc"] = (changed, unpriced,
    replica-disagrees, evaluable). Unpriced rows are dropped from the read and counted (a read needs a 'before')."""
    ship = SCH.ship_ms() if sc_ship_ms == "auto" else sc_ship_ms
    F = F.copy()
    if not len(F) or ship is None:
        F.attrs["sc"] = (0, 0, 0, ship is not None)
        return F
    p, n_ch, n_un, n_dis = SCH.rebase(F[col], F.opened_at, F.pair, F.get("entry_adx_delta"), F.close_reason, ship, sc_delta)
    F[col] = p
    F = F[F[col].notna()].reset_index(drop=True)
    F.attrs["sc"] = (n_ch, n_un, n_dis, True)
    return F


def sc_line(sc, what="Today's exit includes"):
    """the one-line 🔂 re-base status for a scout report. PURE."""
    n_ch, n_un, n_dis, ok = (tuple(sc) + (0, 0, 0, True))[:4] if len(sc) < 4 else sc
    if not ok:
        return ("⚠ 🔂 Second-chance re-base NOT EVALUABLE this run (the DECISION_LOG 532 commit is not in git) — pre-ship fills read at the OLD exit; "
                "do not read this line's verdict.")
    return (f"🔂 {what} the second chance (re-based DL 534): {n_ch} pre-ship fills re-priced from the frozen replica delta"
            + (f" · {n_dis} kept at their booked exit (the replica did not reproduce it — no second-chance credit)" if n_dis else "")
            + (f" · ⚠ {n_un} pre-ship fill(s) the second chance could have changed have no delta — UNPRICED, out of the read" if n_un else "") + ".")


def score_missing(F, now_ms, breadth_fn=None):
    """ML_MKT_HOT for fills RT.attach_mkt_hot left unscored (no engine stamp, past the study cache) — the 🌡 V2 measure (DL 489), budgeted
    public klines for those fills' days only. → (F, complete?)."""
    miss = F.mkt_hot.isna()
    if not miss.any():
        return F, True
    o = pd.to_datetime(F.opened_at[miss].astype(str).str.replace(" ", "T").str[:19], format="ISO8601", errors="coerce")
    if not o.notna().any():
        return F, False
    t_min = int(o.min().value // 10**6)
    if breadth_fn is None:
        closes, complete = MH.fetch_closes(t_min, now_ms, days=sorted(set(o.dropna().dt.normalize())))
        br = MH.breadth_series(closes) if len(closes) and complete else pd.Series(dtype=float)
    else:
        br, complete = breadth_fn(t_min, now_ms)
    sc = MH.score(pd.DataFrame({"o": o.values}), br)
    F = F.copy()
    F.loc[miss, "mkt_hot"] = pd.Series([np.nan if z == "unscored" else (z == "zone") for z in sc.zone], index=F.index[miss], dtype=object)
    return F, complete


def luck_search(D, days, n=BOOT_N, seed=BOOT_SEED):
    """D = per-fill Δ, one column per TP → {column: p} — each column's mean against the BEST of all columns under the same day sign-flips. PURE."""
    M = np.asarray(D, float)
    if not len(M):
        return {c: float("nan") for c in D.columns}
    uniq = list(dict.fromkeys(days))
    di = np.array([uniq.index(x) for x in days])
    S = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n, len(uniq)))[:, di]
    best = (np.einsum("rn,nk->rk", S, M) / len(M)).max(axis=1)
    return {c: (1 + int((best >= M[:, i].mean()).sum())) / (1 + n) for i, c in enumerate(D.columns)}


def verdict(g):
    """g = the line's fills priced under EVERY TP (history + forward) → {TPx: (state, detail)}. PURE."""
    if len(g) < N_MIN:
        return {f"TP{x:g}": ("COLLECTING", f"N {len(g)}/{N_MIN}") for x in TPS}
    D = pd.DataFrame({f"TP{x:g}": (g[f"TP{x:g}"] - g.before).to_numpy(float) for x in TPS})
    ps = luck_search(D, list(g.day))
    hist = g.block.eq("MASTER").to_numpy()
    out = {}
    for x in TPS:
        c = f"TP{x:g}"
        dl = D[c].to_numpy()
        tot = float(dl.sum())
        sh_f = RM.loo(pd.Series(dl), list(g.pair + " " + g.day.str[5:]))["top_share"]
        by_day = pd.Series(dl).groupby(g.day.to_numpy()).sum()
        sh_d = RM.loo(by_day, list(by_day.index))["top_share"]
        s_m, s_n = float(dl[hist].sum()), float(dl[~hist].sum())
        same = s_m != 0 and s_n != 0 and (s_m > 0) == (s_n > 0)
        det = (f"N {len(g)} · avg {g.before.mean():+.2f} → {g[c].mean():+.2f} %/fill · Σ Δ {tot:+.2f} (master {s_m:+.2f} · B25 + forward {s_n:+.2f}) · "
               f"luck p (4-level search) {ps[c]:.2f} · biggest fill {sh_f:.0%} · biggest day {sh_d:.0%} of Δ")
        if tot <= 0:
            out[c] = ("NO EDGE", det)
        elif dl.mean() > 0 and ps[c] <= P_MAX and np.isfinite(sh_f) and sh_f < SHARE_MAX and np.isfinite(sh_d) and sh_d < SHARE_MAX and same:
            out[c] = (CANDIDATE, det)
        else:
            out[c] = ("KEEP OBSERVING", det)
    return out


def _row(label, g, col):
    if not len(g):
        return f"| {label} | 0 | — | — | — | — |"
    if col == "before":
        extra = "—"
    else:
        how = g[col + "_how"]
        extra = (f"{int((how == 'NEW STOP after exit').sum())} new stops · {int((how == 'open at 45 min').sum())} open at 45 min "
                 f"(Σ Δ {float((g[col] - g.before)[how == 'open at 45 min'].sum()):+.2f})")
    return f"| {label} | {len(g)} | {(g[col] > 0).mean():.0%} | {g[col].mean():+.2f} | {g[col].sum():+.2f} | {extra} |"


def run(now_ms=None, paths=None, since_ms=None, breadth_fn=None, master=None, sc_ship_ms="auto", sc_delta=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 479 commit + 10 min splits B25 from FORWARD (not in git → no FORWARD split yet)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 ML_FIXED_TP — momentum longs: fixed take-profit vs today's exit, HISTORY + FORWARD, all fills and kept by ML_MKT_HOT "
         f"(registered 2026-11-05, DECISION_LOG {DL}, OBSERVE only; re-based on the second-chance exit DL 534)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="ML_FIXED_TP watch line")
    since = pd.Timestamp(int(since_ms), unit="ms") if since_ms is not None else None
    F, newest, waiting, ox = cohort(paths, since, master, sc_ship_ms, sc_delta)
    _sc = F.attrs.get("sc", (0, 0, 0, True))
    F, complete = score_missing(F, now_ms, breadth_fn)
    F.attrs["sc"] = _sc
    n = {b: int((F.block == b).sum()) for b in BLOCKS}
    head = (f"Cohort: MASTER {n['MASTER']} · B25 {n['B25']} · FORWARD {n['FORWARD']} full-size momentum longs (FORWARD = opened ≥ {since:%Y-%m-%d %H:%M} UTC)"
            if since is not None else f"Cohort: MASTER {n['MASTER']} · B25 {n['B25']} full-size momentum longs (DL {DL} commit not in git: no FORWARD split yet)")
    L += [head + f" · {waiting} wait for their 45-min post-exit record · {_GK.operator_closed_line(ox)}.",
          sc_line(F.attrs.get("sc", (0, 0, 0, True))) + " Caveat: the TP side still reads the 45 min after the OLD exit, while a second-chance "
          "exit can run to 180 min from entry.", ""]
    if not complete:
        L += [f"⚠ Market-hot read incomplete this run: {int(F.mkt_hot.isna().sum())} fills are unscored for line B (never counted on either side).", ""]
    priced = F[F[[f"TP{x:g}" for x in TPS]].notna().all(axis=1) & F.before.notna()]
    hot = priced.mkt_hot
    is_hot = hot.map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)
    lines = (("A", "all fills", priced), ("B", "kept by ML_MKT_HOT (not hot; unscored out)", priced[hot.notna() & ~is_hot]))
    for key, title, g in lines:
        L += [f"### Line {key} — {title}", "", "| Block · exit | N | WR | avg % | Σ % | after the exit |", "|---|---|---|---|---|---|"]
        for b in BLOCKS + ("ALL",):
            gb = g if b == "ALL" else g[g.block == b]
            if b != "ALL" and not len(gb):
                continue
            L.append(_row(f"**{b}** · today's exit", gb, "before"))
            L += [_row(f"{b} · TP +{x:g} %", gb, f"TP{x:g}") for x in TPS]
        if key == "B":
            L.append(f"| _blocked (hot), all blocks · today's exit_ | {int(is_hot.sum())} | | {priced[is_hot].before.mean():+.2f} | | |"
                     if is_hot.any() else "| _blocked (hot)_ | 0 | | | | |")
            L.append(f"| _unscored — never counted_ | {int(hot.isna().sum())} | | | | |")
        L.append("")
        L += [f"- +{k[2:]} %: **{v[0]}** — {v[1]}" for k, v in verdict(g).items()]
        L.append("")
    L += [f"VERDICT rule (DL {DL}): history + forward together, re-read every run — Σ Δ ≤ 0 → NO EDGE; {CANDIDATE} iff mean Δ > 0 ∧ luck p against the best "
          f"of the 4 levels ≤ {P_MAX:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill and no day ≥ {SHARE_MAX:.0%} of Δ ∧ same sign in MASTER and B25 + FORWARD; "
          "else KEEP OBSERVING. Registration read: reports/ML_FIXED_TP_RECORDED.md.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
