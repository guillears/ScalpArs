#!/usr/bin/env python3
"""🩸🌙 OBS-F1 "LATE-TAPE WEAK SPIKE" — SPIKE_FADE frozen observe line, DECISION_LOG 384 (operator 2026-10-26: "dig into spike fade winners
vs losers by regime" → register). Observe-only: never changes config, never calls the bot API. Reads the orders exports + BTCUSDT 1d
klines (one public GET, weight 5, cached; shared cooldown / used-weight discipline of scout_ml_regime_lines).

REGISTRATION    reports/FADE_REGIME_WL_2026-10-26.md §6 / §8. Master (76 kept fades, stack 2026-10-26a) + B25 (8 closed fades, provisional):
                zone 17 · 59 % · −0.44 %/fill on 14 days / 15 pairs (top day 28 %, top pair 14 %) vs rest 67 · 91 % · +0.42 — expectancy bar
                PASS on ALL (P(avg < 0) 0.978) but FAIL on master alone (12 · 67 % · −0.30, P 0.89, N < 15). Master before → after +$4,237 →
                +$4,494 (+$130 … +$180 after the 30–50 % haircut); B25 fades −$538 → +$136. ⚠ found by an exhaustive within-regime screen
                (in-sample) and the today's-stack BACKTEST refutes it on 3 / 3 seeds (2025 above-EMA200 tape: weak spikes won) — a caveat,
                not a veto (master over backtest), but enough that this is OBSERVE-ONLY (CLAUDE.md OBSERVE-FIRST), not an arm.
ZONE            a SPIKE_FADE fill where ALL hold (🔒 thresholds frozen 2026-10-26, rounded outward from the fitted 48.33 / 1.786, never re-fit):
                  · LATE TAPE: BTCUSDT 1d EMA50 > EMA200 — EMA = ewm(span, adjust=False) on UTC daily closes (≥ 400 bars), read on the last
                    daily bar CLOSED before the fill (bar open + 1 day ≤ opened_at)
                  · entry_pos_di ≤ 48.5          · entry_dist_from_ema13_pct ≤ 1.80
                CONTROL = late tape ∧ scored ∧ not weak (the complement). EARLY tape (EMA50 ≤ EMA200) → outside the line (the zone empties and
                the line PAUSES, never judged on early-tape fills). Missing stamp → UNSCORED (never control); BTC bar missing (cache stale,
                cooldown) → WAITING: the read stays COLLECTING (not persisted) until the bar is there.
COHORT          CLOSED SPIKE_FADE fills opened ≥ 2026-10-26 15:12 UTC = the study's data cutoff (DECISION_LOG 385, operator: "consider all
                this batch too" — every fade the study did not see counts, the rest of B25 included; NOT EVALUABLE until the DL 384
                commit is in git). The B25 fades opened before the cutoff (the study's 8; a fade still OPEN at the 15:11 export would also land here) are
                shown apart (registration block + an in-sample read, never persisted); paper + live POOLED in % (the read is % only; book shown per fill; the
                cohort $ lines are informational and mix books); dedup
                (opened_at[:19], pair, direction), CLOSED wins; operator-closed (MANUAL / MANUAL_* / EXTERNAL_CLOSE) excluded.
🔒 READ          decided ONCE (persisted in reports/.scout_fade_late_weak.json; COLLECTING / HOLD / ERROR never persisted; a paused line
                simply stays COLLECTING), at the first prefix
                by open time where the ZONE reaches N ≥ 15 on ≥ 8 UTC days:
                  zone avg ≥ +0.20 % → RETIRE
                  WR < the forward kept-fade breakeven (|avg loss| / (avg win + |avg loss|) of every kept fade in the prefix; 71 % if < 5 losses)
                  ∧ avg < 0 ∧ day-clustered bootstrap P(avg < 0) ≥ 0.95 (4,000 · seed 7) ∧ ≥ 8 days ∧ no day / pair ≥ 50 % of the zone loss
                  ∧ N ≥ 15 → FILTER_CANDIDATE (OPERATOR CALL; 30–50 % haircut; revert if the first 10 blocked fades re-price avg > 0)
                  else KEEP_OBSERVING.
                HOLD while a fade opened ≤ the crossing fill is still OPEN. Expected read ≈ early December (0.39 zone fills / day in late tape).
Usage           venv/bin/python scripts/scout_fade_late_weak.py      (batch review)
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
import scout_gatekit as _GK                                  # noqa: E402
import scout_fade_watch as _FW                               # noqa: E402  export loader, cohort, HOLD, decide-once, stats
import scout_lite_late_btcfall as _LLB                       # noqa: E402  day-clustered bootstrap + loss-share helpers (DL 353)
import scout_ml_regime_lines as _MRL                         # noqa: E402  budgeted public GET (weight / 418 / 429 / shared cooldown)

DL = 384
KEY = "OBS_F1_LATE_WEAK"
POS_DI_MAX, EXT13_MAX = 48.5, 1.80                           # 🔒 frozen 2026-10-26 (fitted 48.33 / 1.786, rounded outward)
# 🔒 gate cohort floor = the study's data cutoff (DECISION_LOG 385, operator 2026-10-27 "consider all this batch too"): the study read
# B25 from scalpars_orders_paper_2026-10-26_15-11-28.csv (newest fill PARTI 13:19) — every fade opened after it is out-of-sample, so the
# rest of B25 counts. The 8 B25 fades the study saw (5 in the zone) are shown as the REGISTRATION block with an informational
# in-sample read beside the gate read — never persisted, never the decision (they suggested the rule).
FLOOR_PIN = "2026-10-26T15:12:00"
REG_FROM = "2026-10-23T18:30:00"                             # B25 start (bot in PAPER since 10-23 18:30 UTC)
GATE_N, GATE_DAYS, P_MIN, SHARE_MAX, RETIRE_AVG = 15, 8, 0.95, 0.50, 0.20
BE_FALLBACK, BE_MIN_LOSSES = 71.0, 5
BOOT_N, BOOT_SEED = 4000, 7
MIN_DAILY_BARS = 400
DAY_MS = 86_400_000
STAMPS = ("entry_pos_di", "entry_dist_from_ema13_pct")
REVERT = "revert if the first 10 blocked fades re-price average > 0"
STATE = os.path.join(_R, "reports", ".scout_fade_late_weak.json")
KCACHE = os.path.join(_R, "reports", "backtest_cache", "scout_fade_late_weak", "btcusdt_1d.json")
KURL = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=1000"   # weight 5
BLIND = ("Blind spots: the registration pool is master (mixed exit eras re-priced to today's stack) + 8 provisional B25 fades; the zone was "
         "found by an exhaustive within-regime screen (in-sample) and the backtest refutes it on 3 / 3 seeds; late tape is one market state "
         "(≈ 6 weeks since 09-13), so its days are not independent episodes; the EMA200 seed differs from the study's 5m-resampled history by "
         "< 0.2 % weight (a sign flip within ~0.05 % of the crossover can differ).")


def _closed(rows, now_ms):
    """only daily bars already CLOSED at now_ms (open + 1 day ≤ now)."""
    return [r for r in (rows or []) if int(r[0]) + DAY_MS <= int(now_ms)]


def load_btc_1d(now_ms, bars=None, fetch=True):
    """→ DataFrame t (open ms), c (close) of BTCUSDT 1d bars, cached; refetched (one GET) only when the newest CLOSED bar is missing.
    bars = injected rows [[open_ms, o, h, l, c, …], …] (tests). Never raises → (df or None, note)."""
    try:
        rows = bars
        if rows is None:
            try:
                with open(KCACHE) as fh:
                    rows = json.load(fh)
            except Exception:
                rows = None
            need = (int(now_ms) // DAY_MS - 1) * DAY_MS                # open of the newest closed daily bar
            rows = _closed(rows, now_ms)
            have = max((int(r[0]) for r in rows), default=-1) if rows else -1
            if fetch and have < need:
                bud = _MRL.Budget(max_req=1)
                got = _closed(_MRL._get(KURL, bud), now_ms)          # the still-open day bar is never cached as closed
                if got:
                    rows = got
                    try:
                        os.makedirs(os.path.dirname(KCACHE), exist_ok=True)
                        tmp = f"{KCACHE}.{os.getpid()}.tmp"
                        with open(tmp, "w") as fh:
                            json.dump(rows, fh)
                        os.replace(tmp, KCACHE)
                    except Exception:
                        pass
                elif not rows:
                    return None, f"BTC 1d klines unavailable ({bud.stop or '; '.join(bud.notes) or 'no data'})"
        rows = _closed(rows, now_ms)
        if not rows:
            return None, "BTC 1d klines unavailable"
        k = pd.DataFrame([(int(r[0]), float(r[4])) for r in rows], columns=["t", "c"]).drop_duplicates("t").sort_values("t")
        return k.reset_index(drop=True), f"BTC 1d bars {len(k)} (to {pd.Timestamp(int(k.t.iloc[-1]), unit='ms'):%Y-%m-%d})"
    except Exception as e:
        return None, f"BTC 1d klines unreadable ({str(e)[:60]})"


def late_gap(k, opened):
    """→ array of BTC 1d EMA50 / EMA200 − 1 (%) on the last daily bar closed before each fill (NaN if < 400 bars or no bar)."""
    o = pd.to_datetime(pd.Series(opened)).values.astype("datetime64[ms]").astype(np.int64)
    if k is None or not len(k):
        return np.full(len(o), np.nan)
    c = k.c.values.astype(float)
    e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    e200 = pd.Series(c).ewm(span=200, adjust=False).mean().values
    j = np.searchsorted(k.t.values, o - DAY_MS, side="right") - 1      # bar open ≤ T − 1 day ⇔ closed before T
    jj = np.clip(j, 0, None)
    need = (o // DAY_MS - 1) * DAY_MS                                   # the newest bar closed before T (yesterday's UTC day)
    ok = (j >= MIN_DAILY_BARS - 1) & (k.t.values[jj] == need)          # a missing / stale bar → unscored, never an older bar
    return np.where(ok, (e50[jj] / e200[jj] - 1) * 100, np.nan)


def side(df, gap):
    """'zone' / 'control' / 'early' / 'unscored' (stamp missing) / 'nobar' (BTC daily bar missing — transient) per row."""
    pdi = pd.to_numeric(df.entry_pos_di, errors="coerce").values if "entry_pos_di" in df else np.full(len(df), np.nan)
    ext = (pd.to_numeric(df.entry_dist_from_ema13_pct, errors="coerce").values if "entry_dist_from_ema13_pct" in df
           else np.full(len(df), np.nan))
    out = []
    for g, a, b in zip(gap, pdi, ext):
        if pd.isna(a) or pd.isna(b):
            out.append("unscored")
        elif pd.isna(g):
            out.append("nobar")                                         # stamps present, BTC bar missing → wait, never decide
        elif g <= 0:
            out.append("early")
        else:
            out.append("zone" if (a <= POS_DI_MAX and b <= EXT13_MAX) else "control")
    return np.array(out, dtype=object)


def breakeven(y):
    y = pd.Series(y, dtype=float)
    w, l = y[y > 0], y[y < 0]
    if len(l) < BE_MIN_LOSSES or not len(w):
        return BE_FALLBACK, True
    return 100.0 * abs(l.mean()) / (w.mean() + abs(l.mean())), False


def read(g, pending=None):
    """🔒 the frozen read on the forward zone → (verdict, text). g carries side. Never raises."""
    try:
        if not len(g):
            return "COLLECTING", f"⏳ zone 0/{GATE_N} fills on 0/{GATE_DAYS} days"
        g = g.sort_values(["o", "pair"], kind="mergesort").reset_index(drop=True)
        zi = g.index[g.side == "zone"]
        end = None
        for n_, i in enumerate(zi):
            if n_ + 1 >= GATE_N and g.loc[zi[:n_ + 1], "o"].dt.date.nunique() >= GATE_DAYS:
                end = i
                break
        za = g[g.side == "zone"]
        if end is None:
            return "COLLECTING", f"⏳ zone {len(za)}/{GATE_N} fills on {za.o.dt.date.nunique()}/{GATE_DAYS} days — decided once at the read"
        held = _FW._hold(pending, g.o[end])
        if held:
            return "HOLD", held
        pre = g[g.o <= g.o[end]]
        z = pre[pre.side == "zone"]
        be, fb = breakeven(pre.y)
        wr, avg = 100.0 * float((z.y > 0).mean()), float(z.y.mean())
        p = _LLB.p_neg(z.y.values, z.o.dt.date.values, nb=BOOT_N, seed=BOOT_SEED)
        days = z.o.dt.date.nunique()
        dsh, psh = _LLB.loss_share(z, z.o.dt.date), _LLB.loss_share(z, z.pair)
        txt = (f"read at {len(z)} zone fills on {days} days (to {pre.o.iloc[-1]:%m-%d %H:%M}): WR {wr:.0f} % vs breakeven {be:.0f} %"
               f"{' (fallback)' if fb else ''} · avg {avg:+.3f} % · P(avg < 0) {p:.2f} · worst day {dsh:.0%} / pair {psh:.0%} of the loss")
        if avg >= RETIRE_AVG:
            return "RETIRE", txt + f" → RETIRE (zone avg ≥ +{RETIRE_AVG:.2f} %: weak late-tape spikes pay)"
        ok = wr < be and avg < 0 and p >= P_MIN and days >= GATE_DAYS and dsh < SHARE_MAX and psh < SHARE_MAX and len(z) >= GATE_N
        if ok:
            return "FILTER_CANDIDATE", (txt + " → 🚨 FILTER CANDIDATE — OPERATOR CALL (30–50 % in-sample haircut on any Δ; " + REVERT + ")")
        return "KEEP_OBSERVING", txt + " → keep observing (bar not all met; frozen thresholds, never re-fit)"
    except Exception as e:
        return "ERROR", f"read unreadable ({str(e)[:80]})"


def registration_lines(a, k, since, g):
    """the B25 fades the study saw (opened REG_FROM → the cutoff) + an in-sample read of registration ∪ gate cohort — informational,
    NEVER persisted and never the decision."""
    reg_from, reg_to = pd.Timestamp(REG_FROM), min(since, pd.Timestamp(FLOOR_PIN))   # never later than the study cutoff
    if reg_to <= reg_from:
        return ["Registration block: none (the cohort floor is at / before the B25 start)."]
    rf, _ = _FW.forward(a, reg_from)
    rf = rf[rf.o < reg_to]
    rg = rf.assign(side=side(rf, late_gap(k, rf.o))) if len(rf) else rf.assign(side=[])
    L = [f"**Registration block (B25 fades the study saw, {reg_from:%m-%d %H:%M} → {reg_to:%m-%d %H:%M} UTC — in-sample):** "
         f"zone {_FW._stat(rg[rg.side == 'zone'] if len(rg) else rg)} · control {_FW._stat(rg[rg.side == 'control'] if len(rg) else rg)}"
         + "".join(f" · {lab} {int((rg.side == s_).sum())}" for s_, lab in (("early", "early tape"), ("unscored", "unscored"),
                                                                           ("nobar", "waiting (BTC bar missing)")) if len(rg) and (rg.side == s_).any())]
    for r in (rg[rg.side == "zone"].itertuples() if len(rg) else []):
        L.append(f"    · {r.pair} {r.o:%m-%d %H:%M} {'paper' if r.paper else 'live'} · {r.y:+.2f} % · ${r.usd:+.2f}")
    both = pd.concat([rg, g[g.o >= reg_to]] if len(g) else [rg], ignore_index=True)
    if len(both) and (both.side == "nobar").any():
        L.append("  In-sample read skipped this run (BTC daily bar missing).")
    elif len(both):
        v, txt = read(both)
        txt = txt.split(" → ")[0].replace(" — decided once at the read", "")         # no 🚨 / OPERATOR CALL wording: not the gate
        L.append(f"  In-sample read WITH the registration block (B25 only, not master; informational — not the gate, never persisted): would be {v} — {txt}")
    return L


def floor_ms():
    """the pinned study cutoff; None (NOT EVALUABLE) until the DL 384 scout commit is in git. The stamps the line reads predate the
    scout, so fills between the cutoff and the deploy are scored like any later fill."""
    dep = _GK.dl_deploy_ms(DL, lag_min=10, label="OBS-F1 late-tape weak spike")
    return None if dep is None else int(pd.Timestamp(FLOOR_PIN).value // 1_000_000)


def run(now_ms=None, paths=None, since_ms=None, state_path=STATE, bars=None, fetch=True):
    """→ markdown lines (opportunity_scout embeds them; its caller's try catches)."""
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    L = [f"## 🩸🌙 OBS-F1 LATE-TAPE WEAK SPIKE — SPIKE_FADE, DECISION_LOG {DL}, observe-only "
         f"(BTC 1d EMA50 > EMA200 ∧ +DI ≤ {POS_DI_MAX} ∧ Ext% ≤ {EXT13_MAX:.2f})", "",
         "- registration (reports/FADE_REGIME_WL_2026-10-26.md §8): master + B25 zone 17 · 59 % · −0.44 %/fill on 14 days vs rest 67 · 91 % · "
         "+0.42; master alone 12 · 67 % · −0.30 (fails the bar); backtest refutes 3 / 3 seeds → observe-only, OPERATOR CALL at the read."]
    how = "override"
    if since_ms is None:
        since_ms, how = floor_ms(), "the study's data cutoff — every fade the study did not see"
    if since_ms is None:
        L += [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} deploy commit is not in git yet; nothing is counted or persisted.", BLIND, ""]
        return L
    k, knote = load_btc_1d(now_ms, bars=bars, fetch=fetch)
    cur = late_gap(k, [pd.Timestamp(now_ms, unit="ms")])[0] if k is not None else np.nan
    tape = ("unknown" if pd.isna(cur) else (f"LATE (EMA50 / EMA200 {cur:+.2f} %) — line live" if cur > 0
                                            else f"EARLY (EMA50 / EMA200 {cur:+.2f} %) — ⏸ line PAUSED (zone empties, not judged)"))
    L.append(f"- BTC tape now: {tape} · {knote}")
    since = pd.Timestamp(int(since_ms), unit="ms")
    a = _FW.load(paths, extra=STAMPS)
    fw, ox = _FW.forward(a, since)
    pend = _FW.pending_opens(a, since)
    L.append(f"Forward cohort: closed SPIKE_FADE fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({how}) · paper + live · {_FW._stat(fw)} · "
             f"{_GK.operator_closed_line(ox)}")
    g = fw.assign(side=side(fw, late_gap(k, fw.o))) if len(fw) else fw.assign(side=[])
    for s_, lab in (("zone", "zone (late ∧ weak spike) — the block candidate"), ("control", "control (late ∧ strong spike)"),
                    ("early", "early tape (outside the line)"), ("unscored", "unscored (stamp missing)"),
                    ("nobar", "waiting (BTC daily bar missing this run)")):
        L.append(f"  - {lab}: {_FW._stat(g[g.side == s_] if len(g) else g)}")
    for r in (g[g.side == "zone"].itertuples() if len(g) else []):
        L.append(f"    · {r.pair} {r.o:%m-%d %H:%M} {'paper' if r.paper else 'live'} · {r.y:+.2f} % · ${r.usd:+.2f}")
    frozen = _FW._load_state(state_path).get(KEY) if state_path else None
    if len(g) and (g.side == "nobar").any() and not frozen:          # a missing bar must not move the read onto other fills
        v, txt = "COLLECTING", f"⏳ BTC 1d bar missing for {int((g.side == 'nobar').sum())} fill(s) this run — not read, nothing persisted"
    else:
        v, txt = _FW.decide_once(state_path, KEY, *read(g, pend))
    L.append(f"  Read (zone N ≥ {GATE_N} on ≥ {GATE_DAYS} days: avg ≥ +{RETIRE_AVG:.2f} → RETIRE; WR < breakeven ∧ avg < 0 ∧ P(avg < 0) ≥ "
             f"{P_MIN} day-bootstrap ∧ no day / pair ≥ 50 % → FILTER_CANDIDATE; else KEEP_OBSERVING): {v} — {txt}")
    try:                                                               # informational: a failure here must not hide the gate read
        L += registration_lines(a, k, since, g)
    except Exception as e:
        L.append(f"Registration block unavailable ({str(e)[:80]}).")
    L += [BLIND, ""]
    return L


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
