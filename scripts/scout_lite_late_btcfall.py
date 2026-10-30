#!/usr/bin/env python3
"""🕰🧭 Scout — LITE_LATE_BTCFALL observe line (DECISION_LOG 353, operator 2026-10-23: "I consider it a serious filter … add the historic
trades (master batch) as part of the observation"). OBSERVE only — never changes config, never calls the bot API, no network.

WHY             AVAUSDT FRENZY_LITE 10-23 03:00 UTC: a LITE entry 14.9 h after the spike with BTC's 1 h EMA20 slope falling (−0.085) → −3
                stop in 15 min, never positive. Study reports/LITE_LATE_BTCFALL_2026-10-23.md (pre-registered cut, _PREREG.md): two-year
                backtest (yr6 seed 1, today's LITE exit) cell 109 · 50 % · −0.57 %/fill vs rest 1,343 · +0.03 (seeds 2 / 3: −0.61 / −0.60
                below rest); by period 2025 −0.97 · 2026-H1 +0.02 · 2026-H2 −0.36 — FAILED the 2026 out-of-sample check (−0.12, P 0.63) →
                observe line, not a filter. Late alone is NOT bad (late ∧ BTC rising = LITE's best cell); BTC falling alone =
                LITE_BTC1H_FALLING (scout_frenzy_wl_lines.py, DECISION_LOG 288), this cell sits inside it — read the two together.
CELL (frozen)   FRENZY_LITE / FRENZY_LITE_RETRY fill with entry_frenzy_hours ≥ 12.0 ∧ entry_btc_1h_slope ≤ 0. A missing stamp → UNSCORED
                (listed, never counted as rest). Never re-fit.
🔀 SPLIT (operator 2026-10-30, DECISION_LOG 415, no verdict had frozen): LITE and LITE_RETRY are read and decided SEPARATELY, same cell, same gate,
                each with its own frozen state (RETRY behaves oppositely on BTC slope: pooled 7 fills = LITE 4 · +0.87 vs RETRY 3 · −3.00).
COHORT          ① MASTER history (operator): reports/MASTER_POOL_stacked.csv LITE-family rows with stack_keep (today's stack: y = stack_pct,
                $ = stack_pnl) · ② FORWARD: CLOSED LITE-family fills in the orders exports (paper + live via export_glob) whose key
                (opened_at[:19], pair, direction) is not in the master, y = pnl_percentage, $ = pnl. Both: operator-closed (MANUAL /
                MANUAL_* / EXTERNAL_CLOSE) out — scout_gatekit; dedup on the key, CLOSED wins, never id; *_PROBE fills out. Each sleeve's gate reads its own ① + ②
                POOLED (operator override of the study's forward-only cohort). Basis mix: master = stack_pct (today's stack), forward = as
                traded — the verdict is PERSISTED once decided, so a master rebuild can never move it. MASTER MAY ONLY VETO an arm (review:
                a strong master REST or diluted master losing days could otherwise make an arm easier) — see the gate.
🔒 GATE (frozen 2026-10-23, the study's spec; per sleeve since DL 415) — decided ONCE at the first prefix by open time where the cell reaches N ≥ 15 on ≥ 8 UTC days
                (rest = scored non-cell fills opened ≤ the prefix end):
                  RETIRE                     if cell avg ≥ rest avg;
                  ARM CANDIDATE (operator    iff cell WR < LITE breakeven (|avg loss| / (avg win + |avg loss|) of the prefix's fills of that sleeve
                  picks block or ½ size)         fills when N ≥ 30, else 62 % — TP +2 vs −3 with fees) ∧ day-clustered bootstrap
                                                 P(cell avg < 0) ≥ 0.95 (4,000, seed 7) ∧ no single day or pair ≥ 50 % of the cell's gross loss;
                                             AND, so the master can only veto, the same on the FORWARD fills of the prefix alone: forward
                                             cell avg < forward rest avg (when forward rest exists) ∧ no single forward day or pair ≥ 50 % of
                                             the forward cell's gross loss;
                  else KEEP OBSERVING.
                An armed rule ships with the study's revert gate: back off if the first 15 post-arm blocked-cell shadow fills average > the
                kept LITE fills of the same period.
Usage           venv/bin/python scripts/scout_lite_late_btcfall.py      (batch review)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402

DL = 353
LITE_FAMILY = ("FRENZY_LITE", "FRENZY_LITE_RETRY")
LATE_H, SLOPE_MAX = 12.0, 0.0                                # 🔒 frozen cell
GATE_N, GATE_DAYS, P_MIN, SHARE_MAX = 15, 8, 0.95, 0.50
BE_FALLBACK, BE_MIN_N = 62.0, 30
BOOT_N, BOOT_SEED = 4000, 7
MASTER = os.path.join(_R, "reports", "MASTER_POOL_stacked.csv")
STATE = os.path.join(_R, "reports", ".lite_late_btcfall_gate.json")      # 🔒 base name: one frozen verdict per sleeve (state_for), written once at the read
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
        "entry_frenzy_hours", "entry_btc_1h_slope", "frenzy_lite_tp_used", "stack_keep", "stack_pct", "stack_pnl", "era")
BACKTEST_REF = ("backtest ref (yr6 seed 1, today's LITE exit): cell 109 · 50 % · −0.57 %/fill vs rest 1,343 · +0.03 · 2025 −0.97 / 2026-H1 "
                "+0.02 / 2026-H2 −0.36 · 2026 out-of-sample −0.12 (P 0.63, failed) — reports/LITE_LATE_BTCFALL_2026-10-23.md")
KNOWN_UNEXPORTED = ("AVAUSDT 10-23 03:00 FRENZY_LITE · 14.9 h · slope −0.085 · −2.85 % (−$53.99) — the motivating case; its batch was reset "
                    "unsaved, so it is listed here and never counted")


def _ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace(" ", "T"), errors="coerce", format="%Y-%m-%dT%H:%M:%S")


def _key(df):
    return df.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + df.pair.astype(str) + "|" + df.direction.astype(str).str.upper()


def _read(paths):
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS))
    a = pd.concat(fr, ignore_index=True)
    for c in COLS:
        if c not in a:
            a[c] = None
    return a


def _shape(a, src, y, usd):
    a = a[~a.entry_strategy.astype(str).str.endswith("_PROBE")]
    a = a.assign(key=_key(a), src=src, o=_ts(a.opened_at).values, y=pd.to_numeric(a[y], errors="coerce"),
                 usd=pd.to_numeric(a[usd], errors="coerce"), h=pd.to_numeric(a.entry_frenzy_hours, errors="coerce"),
                 s=pd.to_numeric(a.entry_btc_1h_slope, errors="coerce"),
                 tp=pd.to_numeric(a.frenzy_lite_tp_used, errors="coerce"))
    return a[a.o.notna() & a.y.notna()]


def load_master(path=MASTER):
    """master LITE-family rows kept by today's stack → (kept, operator-closed, ALL master keys). y = stack_pct (today's stack). The key set
    covers every master row (any strategy, stack-blocked and operator-closed included) so none of them re-enters as a forward fill."""
    a = _read([path]) if path and os.path.exists(path) else _read([])
    keys = set(_key(a)) if len(a) else set()
    a = a[a.entry_strategy.astype(str).isin(LITE_FAMILY) & a.stack_keep.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))]
    a = a.assign(_k=_key(a)).drop_duplicates("_k", keep="last").drop(columns=["_k"]) if len(a) else a
    kept, ox = _GK.split_operator_closed(a) if len(a) else (a, a)
    return _shape(kept, "master", "stack_pct", "stack_pnl"), ox, keys


def load_forward(master_keys, paths=None):
    """CLOSED LITE-family export fills not in the master → (kept, operator-closed). Dedup on the key, CLOSED wins, then the newest export."""
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    a = _read(paths)
    if not len(a):
        return _shape(a, "forward", "pnl_percentage", "pnl"), a
    a = a.assign(_k=_key(a), _c=(a.status.astype(str) == "CLOSED").astype(int)).reset_index(drop=True)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates("_k", keep="last")
    a = a[(a.status.astype(str) == "CLOSED") & a.entry_strategy.astype(str).isin(LITE_FAMILY) & ~a._k.isin(set(master_keys))]
    kept, ox = _GK.split_operator_closed(a.drop(columns=["_k", "_c", "_i"])) if len(a) else (a, a)
    return _shape(kept, "forward", "pnl_percentage", "pnl"), ox


def classify(df):
    """→ 'cell' / 'rest' / 'unscored' per row (frozen cut; a missing stamp is unscored, never rest)."""
    sc = df.h.notna() & df.s.notna()
    return np.where(~sc, "unscored", np.where((df.h >= LATE_H) & (df.s <= SLOPE_MAX), "cell", "rest"))


def breakeven(y):
    w, l = y[y > 0], y[y < 0]
    if len(y) < BE_MIN_N or not len(w) or not len(l):
        return BE_FALLBACK
    return 100.0 * abs(l.mean()) / (w.mean() + abs(l.mean()))


def p_neg(vals, days, nb=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap P(per-fill mean < 0)."""
    v, dd = np.asarray(vals, float), np.asarray(days)
    groups = [v[dd == x] for x in pd.unique(dd)]
    if not groups:
        return np.nan
    s = np.array([g.sum() for g in groups]); n = np.array([len(g) for g in groups])
    idx = np.random.default_rng(seed).integers(0, len(groups), size=(nb, len(groups)))
    return float(((s[idx].sum(1) / n[idx].sum(1)) < 0).mean())


def loss_share(cell, by):
    loss = cell[cell.y < 0]
    tot = -loss.y.sum()
    return float((-loss.groupby(by).y.sum()).max() / tot) if tot > 0 else 0.0


def gate(pool):
    """🔒 the frozen read on the pooled scored fills (sorted by open time) → (verdict, text). Never raises."""
    try:
        sc = pool[pool.grp != "unscored"].sort_values(["o", "pair", "direction", "src"], kind="mergesort").reset_index(drop=True)
        cell_idx = sc.index[sc.grp == "cell"]
        end = None
        for k, i in enumerate(cell_idx):
            if k + 1 >= GATE_N and sc.loc[cell_idx[:k + 1], "o"].dt.date.nunique() >= GATE_DAYS:
                end = i
                break
        c_all = sc[sc.grp == "cell"]
        if end is None:
            return "COLLECTING", (f"⏳ cell {len(c_all)}/{GATE_N} fills on {c_all.o.dt.date.nunique()}/{GATE_DAYS} days "
                                  "(master + forward pooled) — decided once at the read")
        pre = sc[sc.o <= sc.o[end]]                                  # every scored fill opened by the crossing fill's time
        cell, rest = pre[pre.grp == "cell"], pre[pre.grp == "rest"]
        ca, ra = float(cell.y.mean()), float(rest.y.mean()) if len(rest) else np.nan
        head = (f"read at {len(cell)} cell fills on {cell.o.dt.date.nunique()} days (to {pre.o.iloc[-1]:%m-%d %H:%M}): cell avg {ca:+.3f} % vs "
                f"rest {ra:+.3f} % ({len(rest)})")
        if len(rest) and ca >= ra:
            return "RETIRE", head + " → cell ≥ rest: retire the line (frozen, never re-fit)"
        be = breakeven(pre.y)
        wr = 100.0 * float((cell.y > 0).mean())
        p = p_neg(cell.y.values, cell.o.dt.date.values)
        dsh, psh = loss_share(cell, cell.o.dt.date), loss_share(cell, cell.pair)
        fc, fr = cell[cell.src == "forward"], rest[rest.src == "forward"]      # the master may only VETO: the forward fills must agree alone
        f_cmp = (not len(fr)) or (len(fc) and float(fc.y.mean()) < float(fr.y.mean()))
        fdsh, fpsh = (loss_share(fc, fc.o.dt.date), loss_share(fc, fc.pair)) if len(fc) else (0.0, 0.0)
        ok = (wr < be and p >= P_MIN and dsh < SHARE_MAX and psh < SHARE_MAX and len(fc) > 0 and bool(f_cmp)
              and fdsh < SHARE_MAX and fpsh < SHARE_MAX)
        txt = (f" · WR {wr:.0f} % vs breakeven {be:.0f} % · P(avg < 0) {p:.2f} · worst day {dsh:.0%} / pair {psh:.0%} of the loss · "
               f"forward-only: cell {len(fc)}" + (f" avg {fc.y.mean():+.3f}" if len(fc) else "") + f" vs rest {len(fr)}"
               + (f" avg {fr.y.mean():+.3f}" if len(fr) else "") + f" · worst forward day {fdsh:.0%} / pair {fpsh:.0%}")
        if ok:
            return "ARM_CANDIDATE", head + txt + " → 🚨 ARM CANDIDATE: operator picks block or ½ size (+ the pre-committed revert gate)"
        return "KEEP_OBSERVING", head + txt + " → keep observing (bars not all met; never re-fit)"
    except Exception as e:
        return "ERROR", f"gate unreadable ({str(e)[:80]})"


def _line(name, g):
    if not len(g):
        return f"- {name}: N 0"
    return (f"- {name}: N {len(g)} · {g.o.dt.date.nunique()} days · WR {(g.y > 0).mean():.0%} · avg {g.y.mean():+.3f} % · "
            f"Σ {g.y.sum():+.2f} pts · ${g.usd.sum(skipna=True):+.2f}")


def frozen_verdict(state_path, v, txt):
    """🔒 decided ONCE: the first non-COLLECTING verdict is written to state_path and returned on every later run (a master rebuild or a
    backfilled export can never move it). → (verdict, text, frozen_now). Never raises; an unwritable path just isn't persisted."""
    try:
        if state_path and os.path.exists(state_path):
            with open(state_path) as fh:
                st = json.load(fh)
            return st["verdict"], st["text"] + f" (frozen {st['at'][:16]} UTC)", False
        if state_path and v not in ("COLLECTING", "ERROR"):
            with open(state_path, "w") as fh:
                json.dump({"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}, fh)
            return v, txt, True
    except Exception:
        pass
    return v, txt, False


def state_for(state_path, es):
    """the sleeve's own frozen-verdict file (DL 415). None stays None (no write)."""
    return state_path.replace(".json", f"_{es}.json") if state_path else None


def run(now_ms=None, paths=None, master_path=MASTER, state_path=STATE):
    L = [f"## 🕰🧭 LITE_LATE_BTCFALL — LITE ≥ {LATE_H:.0f} h after the spike ∧ BTC 1 h slope ≤ 0 (DECISION_LOG {DL}, observe-only, frozen gate)", ""]
    m, mox, mkeys = load_master(master_path)
    f, fox = load_forward(mkeys, paths)
    pool = pd.concat([m, f], ignore_index=True)
    pool["grp"] = classify(pool) if len(pool) else []
    L.append(f"Cohort: LITE + LITE_RETRY · master (today's stack) {len(m)} + forward exports {len(f)} · operator-closed out "
             f"(master {len(mox)}, forward {len(fox)})")
    for es in LITE_FAMILY:                                           # 🔀 (415) each sleeve read and decided on its own
        p_s = pool[pool.entry_strategy.astype(str) == es] if len(pool) else pool
        L.append(f"### {es}")
        for src, lab in (("master", "MASTER history"), ("forward", "FORWARD (exports, not in master)"), (None, "master + forward (the gate reads this)")):
            g = p_s if src is None else (p_s[p_s.src == src] if len(p_s) else p_s)
            L.append(f"{lab}:")
            for grp, name in (("cell", "cell (late ∧ BTC falling)"), ("rest", "rest of the sleeve"), ("unscored", "unscored (stamp missing)")):
                L.append("  " + _line(name, g[g.grp == grp] if len(g) else g))
        v, txt, _ = frozen_verdict(state_for(state_path, es), *gate(p_s))
        L.append(f"Gate verdict: {v} — {txt}")
    c = pool[pool.grp == "cell"].sort_values("o") if len(pool) else pool
    if len(c):
        L.append("Cell fills: " + " · ".join(f"{r.pair} {r.o:%m-%d %H:%M} {r.entry_strategy} [{r.src}] {r.h:.1f} h slope {r.s:+.3f} "
                                              f"TP {r.tp if r.tp == r.tp else '?'} → {r.y:+.2f} %" for r in c.itertuples()))
    L.append("Not counted: " + KNOWN_UNEXPORTED)
    L.append(BACKTEST_REF)
    L.append("Read with LITE_BTC1H_FALLING (scripts/scout_frenzy_wl_lines.py, DECISION_LOG 288) — this cell sits inside it.")
    return L + [""]


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
