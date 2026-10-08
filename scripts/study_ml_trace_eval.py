#!/usr/bin/env python3
"""MOMENTUM-LONG recall trace (2026-10-08) — step 2: per-signal classification of every live-only and replay-only ML fill.

Read-only. Inputs: reports/study_ml_trace_matchset.csv (step 1) and the decision journals + orders of three replay views over the
same days (all real-engine replays, scripts/engine_replay.py):
  yr5  = TODAY's frozen config (181131e), synthetic live cadence 111.6 ± 5 s with a random phase per seed, fixed harness, 3 seeds
  asw  = the config AS IT WAS at each moment (config_history_v2), at LIVE's scan clock (livephase_scans_v2), fixed harness
  lp   = (near-)today's config (frozen Sep-28 live config; lp_b1316m = yr4 frozen config) at LIVE's scan clock (pre-Oct-4 harness
         except lp_b1316m)
Journal lines used: SCAN (BTC readings the engine used), FAILS (every failing gate of a pair at a scan, MOMENTUM src), BLOCK (the
counted gate + the engine's pair indicator ctx), OPEN, ADMIT. Capacity blocks (BOOK_FULL / PAIR_HELD / COOLDOWN / NO_BALANCE / caps)
are counted by pair when tagged; otherwise the replay's own open book (its orders) is used.

LIVE-ONLY (live full-size ML fill, no yr5 ML fill same pair ±10 min) — per seed:
  TOOK_OTHER_TIME (same pair ML 10 min–3 h away) · PAIR_HELD (replay held the pair in any sleeve) · CAPACITY (replay book at
  max_open or a capacity block near the decision) · GATE:<gate> (the replay's FAILS line for the pair nearest live's decision
  second; all gates listed) · DIR_SHORT (the replay read the pair as a SHORT candidate at that scan) · NO_CANDIDATE (pair scanned,
  no LONG line) · NOT_SCANNED (no line for the pair within ±10 min).
  Data source of the replay's forming candle: ticks_q pair-day present and older than the chunk's run start → TICKS, else 1M.
REPLAY-ONLY (yr5 ML fill while live was up, no live full-size ML fill same pair ±10 min):
  LIVE_PROBE (live took it as a 1× probe) · LIVE_PAIR_HELD · LIVE_SLOTS_FULL (live open bot positions ≥ as-was max_open) ·
  LIVE_OTHER_TIME (live ML same pair ≤ 3 h) · then the live-clock replays decide: ASW_TOOK (as-was rules at live's clock take it →
  live-side miss) · RULE_THEN:<gate> (as-was blocks it, today's-config live-clock replay lp takes it → genuine rule-era difference)
  · PHASE:<gate> (both live-clock replays block it → yr5's synthetic scan phase caught a knife-edge pass) · live journal BLOCK
  (Sep-28 →) when present.
Outputs: reports/study_ml_trace_seedrows.csv (live-only × seed), reports/study_ml_trace_extras.csv (replay-only), pickle cache of
the parsed journals in the scratch dir passed by --cache (default reports/backtest_cache/study_ml_trace_journal.pkl).
Usage: venv/bin/python scripts/study_ml_trace_eval.py [--cache PATH] [--procs 6]
"""
import argparse, glob, json, os, pickle, re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import yr5_fills_trimmed as YT                                          # noqa: E402

REP = os.path.join(ROOT, "reports")
RP = os.path.join(REP, "backtest_cache", "replay")
JD = os.path.join(RP, "year", "journal")
TQ = os.path.join(REP, "backtest_cache", "ticks_q")
W0, W1 = pd.Timestamp("2026-06-15"), pd.Timestamp("2026-10-04")
ASW = ["asw_base1m", "asw_base2", "asw_b1a", "asw_b1b", "asw_b2", "asw_b3m", "asw_b45", "asw_b67m", "asw_b812m", "asw_b1316m"]
LP = ["lp_base1", "lp_base2", "lp_b1a", "lp_b1b", "lp_b2", "lp_b3", "lp_b45", "lp_b612", "lp_b1316m"]
CAP = ("BOOK_FULL", "PAIR_HELD", "COOLDOWN", "NO_BALANCE", "WILLY_HOLD", "GLOBAL_HOLD", "GROSS_CAP_SKIP", "LIQ_CAP_SKIP",
       "BRACKET_CAP_SKIP", "SLOTS_FULL", "MAX_OPEN")
RX_PAIR = re.compile(r'"pair":"([^"]+)"')
DEC_OFF = pd.Timedelta(seconds=8)          # live decision ≈ opened_at − 8 s (6 s open path + latency; maker fills open later)


def parse_dir(args):
    tag, pairs, a, z = args
    rows, scans = [], []
    for f in sorted(glob.glob(os.path.join(JD, tag, "decisions-*.jsonl"))):
        for line in open(f):
            if '"e":"SCAN"' in line:
                q = json.loads(line)
                scans.append((q["t"], q.get("btc_rsi"), q.get("btc_adx"), q.get("btc_slope"), q.get("veto_long") or ""))
                continue
            m = RX_PAIR.search(line)
            if m is None:
                if '"e":"BLOCK"' in line and any(g in line for g in CAP):
                    q = json.loads(line)
                    rows.append((q["t"], "", q["e"], q.get("dir", ""), q.get("gate", ""), "", None))
                continue
            if m.group(1) not in pairs:
                continue
            q = json.loads(line)
            e = q.get("e")
            if e not in ("FAILS", "BLOCK", "OPEN", "ADMIT"):
                continue
            g = q.get("gates") if e == "FAILS" else (q.get("gate") or q.get("strategy") or "")
            ctx = q.get("ctx")
            if e == "OPEN":
                ctx = {"strategy": q.get("strategy"), "cell": q.get("cell"), "price": q.get("price")}
            rows.append((q["t"], q["pair"], e, q.get("dir", ""), g or "", q.get("src", ""), json.dumps(ctx) if ctx else None))
    R = pd.DataFrame(rows, columns=["t", "pair", "e", "dir", "gate", "src", "ctx"])
    S = pd.DataFrame(scans, columns=["t", "btc_rsi", "btc_adx", "btc_slope", "veto_long"])
    for d in (R, S):
        d["t"] = pd.to_datetime(d.t.astype(str).str[:23], format="mixed")
    R = R[(R.t >= a) & (R.t < z)]; S = S[(S.t >= a) & (S.t < z)]
    return tag, R.reset_index(drop=True), S.reset_index(drop=True)


def meta_win(tag):
    m = json.load(open(os.path.join(RP, f"{tag}_meta.json")))
    return pd.Timestamp(m["start_ms"], unit="ms"), pd.Timestamp(m["end_ms"], unit="ms"), m


def run_orders(tag, trim=True):
    a, z, _ = meta_win(tag)
    o = pd.read_csv(os.path.join(RP, f"{tag}_orders.csv"), low_memory=False)
    o = o[o.status == "CLOSED"].copy()
    o["t"] = pd.to_datetime(o.opened_at.astype(str).str[:19].str.replace("T", " "))
    o["tc"] = pd.to_datetime(o.closed_at.astype(str).str[:19].str.replace("T", " "), errors="coerce")
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    o["sleeve"] = [YT.sleeve_of(s, d) for s, d in zip(o.entry_strategy, o.direction)]
    o["probe"] = o.cell_multiplier_source.fillna("").astype(str).str.contains("PROBE")
    o["tag"] = tag
    if trim:
        o = o[(o.t >= a) & (o.t < z)]
    return o


def load_all(pairs, cache, procs):
    if os.path.exists(cache):
        J = pickle.load(open(cache, "rb"))
        if J.get("pairs") == sorted(pairs):
            return J
    jobs = []
    for tag, chunk, seed, a, z, _w in YT.chunk_windows("yr5"):
        if z > W0 and a < W1:
            jobs.append((tag, set(pairs), a, z))
    for tag in ASW + LP:
        a, z, _ = meta_win(tag)
        jobs.append((tag, set(pairs), a, z))
    with Pool(procs) as P:
        res = P.map(parse_dir, jobs)
    J = {"pairs": sorted(pairs), "rows": {}, "scans": {}}
    for tag, R, S in res:
        J["rows"][tag] = R; J["scans"][tag] = S
    pickle.dump(J, open(cache, "wb"))
    return J


def cat(gates):
    """family of a FAILS gate string (first gate decides the family order below)."""
    gs = [g.split("[")[0].replace("MACRO:", "") for g in gates.split("+") if g]
    fam = []
    for g in gs:
        if g in ("PAIR_RSI_MOMENTUM_LOADX", "LONG_MEGACAP_BLOCK", "LONG_HEAT_BLOCK", "LONG_CHOP_BURST") or g.startswith("CALM3D"):
            fam.append("TODAY_RULE")
        elif g in ("BTC_ACCEL_CHASE_LONG", "LONG_UNMATCHED_ONLY", "COOLDOWN") or g.startswith("RSICEIL_DOOR"):
            fam.append("STATE_ROUTING")
        elif g.startswith("BTC") or g.startswith("LONG_BTC") or g in ("ADX_DELTA_BTC_ADX_CROSS", "FAN_RATIO_GATE"):
            fam.append("BTC_MARKET")
        else:
            fam.append("PAIR_CANDLE")
    for f in ("TODAY_RULE", "STATE_ROUTING", "BTC_MARKET", "PAIR_CANDLE"):
        if f in fam:
            return f, gs
    return "PAIR_CANDLE", gs


def tick_src(pair, t, run_start):
    fp = os.path.join(TQ, pair, f"{t:%Y-%m-%d}.npz")
    if not os.path.exists(fp):
        return "NO_TICKS"
    return "TICKS" if os.path.getmtime(fp) < run_start else "TICKS_LATER"


def nearest_lines(R, pair, td, lo=-330, hi=90):
    d = R[(R.pair == pair)]
    if not len(d):
        return d
    dt = (d.t - td).dt.total_seconds()
    return d[(dt >= lo) & (dt <= hi)].assign(dts=dt[(dt >= lo) & (dt <= hi)])


def classify_live(x, seed, R, S, O_all, maxopen, run_start):
    td = x.t - DEC_OFF
    o = O_all[(O_all.pair == x.pair) & (O_all.sleeve == "MOM-long") & ~O_all.probe & ((O_all.t - x.t).abs() <= pd.Timedelta(hours=3))]
    held = O_all[(O_all.pair == x.pair) & (O_all.t <= td) & ((O_all.tc > td) | O_all.tc.isna())]
    out = dict(seed=seed, src=tick_src(x.pair, x.t, run_start))
    sc = S[(S.t - td).abs() <= pd.Timedelta(seconds=150)]
    if len(sc):
        s = sc.iloc[((sc.t - td).abs()).argmin()]
        out.update(scan_dt_s=(s.t - td).total_seconds(), rep_btc_rsi=s.btc_rsi, rep_btc_adx=s.btc_adx, rep_veto_long=s.veto_long)
    if len(held):
        h = held.iloc[0]
        return dict(out, cls="PAIR_HELD", detail=f"{h.sleeve} {h.entry_strategy} opened {h.t:%m-%d %H:%M}")
    if len(o):
        return dict(out, cls="TOOK_OTHER_TIME", detail=f"{(o.t - x.t).dt.total_seconds().abs().min() / 60:.0f} min away",
                    other_pct=o.iloc[((o.t - x.t).abs()).argmin()].pct)
    n_open = len(O_all[(O_all.t <= td) & ((O_all.tc > td) | O_all.tc.isna()) & ~O_all.probe])
    L = nearest_lines(R, x.pair, td)
    capb = R[(R.pair == "") & ((R.t - td).dt.total_seconds().abs() <= 60)]
    lng = L[(L.dir.isin(["LONG", "", "ANY"])) & (L.e.isin(["FAILS", "BLOCK"]))]
    capl = lng[lng.gate.isin(CAP)]
    fl = lng[(lng.e == "FAILS") & (lng.src.isin(["MOMENTUM", ""]))]
    if len(capl) or n_open >= maxopen:
        return dict(out, cls="CAPACITY", detail=f"{n_open}/{maxopen} open; " + ",".join(sorted(set(capl.gate))))
    if len(fl):
        f = fl.iloc[(fl.dts.abs()).argmin()]
        fam, gs = cat(f.gate)
        bl = lng[(lng.e == "BLOCK") & lng.ctx.notna()]
        ctx = json.loads(bl.iloc[(bl.dts - f.dts).abs().argmin()].ctx) if len(bl) else {}
        out.update(fails_dt_s=f.dts, gates=f.gate, n_fails_lines=len(fl),
                   all_fail_scans=int((fl.groupby(fl.t.dt.floor("30s")).size() > 0).sum()),
                   rep_rsi=ctx.get("rsi"), rep_adx=ctx.get("adx"), rep_ema5=ctx.get("ema5"), rep_ema13=ctx.get("ema13"),
                   rep_ema20=ctx.get("ema20"), rep_ema50=ctx.get("ema50"), rep_price=ctx.get("price"))
        return dict(out, cls=f"GATE:{fam}", detail=gs[0] if len(gs) == 1 else "+".join(gs[:4]))
    sh = L[(L.dir == "SHORT") & (L.e == "FAILS")]
    if len(sh):
        return dict(out, cls="DIR_SHORT", detail=sh.iloc[(sh.dts.abs()).argmin()].gate[:80])
    if len(capb) and n_open >= maxopen - 1:
        return dict(out, cls="CAPACITY", detail=f"{n_open}/{maxopen} open; pairless " + ",".join(sorted(set(capb.gate))))
    anyl = R[(R.pair == x.pair) & ((R.t - td).dt.total_seconds().abs() <= 600)]
    if len(anyl):
        return dict(out, cls="NO_CANDIDATE", detail=f"{len(anyl)} lines ±10 min, none LONG near")
    return dict(out, cls="NOT_SCANNED", detail="no journal line for the pair ±10 min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(REP, "backtest_cache", "study_ml_trace_journal.pkl"))
    ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    M = pd.read_csv(os.path.join(REP, "study_ml_trace_matchset.csv"), low_memory=False)
    M["t"] = pd.to_datetime(M.t)
    LV = M[(M.side == "LIVE")].copy()
    RE = M[(M.side == "REPLAY")].copy()
    for c in ("matched_live", "live_up"):
        RE[c] = RE[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    pairs = sorted(set(LV.pair) | set(RE[RE.live_up].pair))
    J = load_all(pairs, A.cache, A.procs)
    cfg = json.load(open(os.path.join(RP, "frozen_config_yr5_181131e.json")))
    maxopen = int(cfg.get("investment", {}).get("max_open_positions", 4))
    # ── yr5 per seed: orders (all sleeves, untrimmed for book state) + journals ──
    chunks = [(tag, seed, a, z) for tag, chunk, seed, a, z, _w in YT.chunk_windows("yr5") if z > W0 and a < W1]
    OR = {tag: run_orders(tag, trim=False) for tag, *_ in chunks}
    RS = {}
    for tag, seed, a, z in chunks:
        m = json.load(open(os.path.join(RP, f"{tag}_meta.json")))
        RS[tag] = os.path.getmtime(os.path.join(RP, f"{tag}_orders.csv")) - float(m.get("elapsed_s", 0)) - 600
    rows = []
    for i, x in LV[LV.t >= W0].iterrows():
        for tag, seed, a, z in chunks:
            if not (a <= x.t < z):
                continue
            hit = isinstance(x.m_seeds, str) and str(seed) in x.m_seeds.split(",")
            base = dict(live_idx=i, pair=x.pair, t=x.t, era=x.era, stack_keep=x.stack_keep, stack_reason=x.stack_reason,
                        in_master=x.in_master, pct_live=x.pct_live_raw, pct_live_stack=x.pct_live_stack, tag=tag)
            if hit:
                rows.append(dict(base, seed=seed, cls="MATCHED", src=tick_src(x.pair, x.t, RS[tag])))
                continue
            rows.append(dict(base, **classify_live(x, seed, J["rows"][tag], J["scans"][tag], OR[tag], maxopen, RS[tag])))
    SR = pd.DataFrame(rows)
    # live stamps for Δ (BTC + pair)
    for c in ("entry_btc_rsi", "entry_btc_adx", "entry_rsi", "entry_adx", "entry_price"):
        SR["live_" + c[6:]] = pd.to_numeric(SR.live_idx.map(LV[c]), errors="coerce")
    SR["d_btc_rsi"] = pd.to_numeric(SR.get("rep_btc_rsi"), errors="coerce") - SR.live_btc_rsi
    SR["d_btc_adx"] = pd.to_numeric(SR.get("rep_btc_adx"), errors="coerce") - SR.live_btc_adx
    SR["d_rsi"] = pd.to_numeric(SR.get("rep_rsi"), errors="coerce") - SR.live_rsi
    SR["d_adx"] = pd.to_numeric(SR.get("rep_adx"), errors="coerce") - SR.live_adx
    SR["d_price_pct"] = (pd.to_numeric(SR.get("rep_price"), errors="coerce") / SR.live_price - 1) * 100
    SR.to_csv(os.path.join(REP, "study_ml_trace_seedrows.csv"), index=False)

    # ── replay-only (live up) ──
    LALL = pd.concat([pd.read_csv(os.path.join(REP, "COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv"), low_memory=False)
                      .query("opened_at < '2026-07-11'"),
                      pd.read_csv(os.path.join(REP, "BATCH1_2026-07-11to31_orders_FINAL.csv"), low_memory=False),
                      pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False).query("era not in ['BASE','B1']")],
                     ignore_index=True)
    LALL = LALL[LALL.status == "CLOSED"].copy()
    LALL["t"] = pd.to_datetime(LALL.opened_at.astype(str).str[:19].str.replace("T", " "))
    LALL["tc"] = pd.to_datetime(LALL.closed_at.astype(str).str[:19].str.replace("T", " "), errors="coerce")
    LALL = LALL.drop_duplicates(["t", "pair", "direction"])
    LALL["probe"] = LALL.cell_multiplier_source.fillna("").astype(str).str.contains("PROBE")
    LALL["ml"] = (LALL.entry_strategy.fillna("MOMENTUM").astype(str) == "MOMENTUM") & (LALL.direction == "LONG")
    LALL = LALL[LALL.entry_strategy.fillna("").astype(str) != "MANUAL"]
    # as-was max_open from config history
    CH = sorted((int(os.path.basename(f).split("_")[0]), f) for f in glob.glob(os.path.join(RP, "config_history_v2", "*_*.json"))
                if os.path.basename(f).split("_")[0].isdigit())
    _mo = {}

    def mo_at(t):
        ms = int(pd.Timestamp(t).value // 10**6)
        k = max(0, int(np.searchsorted([c[0] for c in CH], ms, side="right")) - 1)
        if k not in _mo:
            _mo[k] = int(json.load(open(CH[k][1])).get("investment", {}).get("max_open_positions", 4))
        return _mo[k]
    AO = pd.concat([run_orders(t) for t in ASW], ignore_index=True)
    PO = pd.concat([run_orders(t) for t in LP], ignore_index=True)
    AWIN = {t: meta_win(t)[:2] for t in ASW}; PWIN = {t: meta_win(t)[:2] for t in LP}
    # live journal (Sep-28 →)
    lj = []
    for f in sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"))):
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("t", "e", "pair", "dir", "gate", "gates", "src"), low_memory=False)
        except Exception:
            continue
        lj.append(d[d.e.isin(["BLOCK", "FAILS", "OPEN", "SCAN"])])
    LJ = pd.concat(lj).drop_duplicates() if lj else pd.DataFrame(columns=["t", "e", "pair", "dir", "gate"])
    LJ["t"] = pd.to_datetime(LJ.t.astype(str).str[:23], format="mixed", errors="coerce")
    lj0 = LJ[LJ.e == "SCAN"].t.min() if len(LJ) else pd.NaT
    ex = []
    for i, y in RE[RE.live_up & ~RE.matched_live & (RE.t >= W0)].iterrows():
        td = y.t - DEC_OFF
        d = dict(rep_idx=i, pair=y.pair, t=y.t, seed=y.seed, pct_rep=y.pct_rep, cell=y.cell, rep_reason=y.rep_reason)
        lp_ = LALL[(LALL.pair == y.pair) & ((LALL.t - y.t).abs() <= pd.Timedelta(minutes=10)) & LALL.ml & LALL.probe]
        held = LALL[(LALL.pair == y.pair) & (LALL.t <= td) & ((LALL.tc > td) | LALL.tc.isna())]
        oth = LALL[(LALL.pair == y.pair) & LALL.ml & ~LALL.probe & ((LALL.t - y.t).abs() <= pd.Timedelta(hours=3))]
        nopen = int(((LALL.t <= td) & ((LALL.tc > td) | LALL.tc.isna()) & ~LALL.probe).sum())
        a_t = [t for t, (a, z) in AWIN.items() if a <= y.t < z]; p_t = [t for t, (a, z) in PWIN.items() if a <= y.t < z]
        am = AO[(AO.pair == y.pair) & (AO.sleeve == "MOM-long") & ((AO.t - y.t).abs() <= pd.Timedelta(minutes=10))]
        pm = PO[(PO.pair == y.pair) & (PO.sleeve == "MOM-long") & ((PO.t - y.t).abs() <= pd.Timedelta(minutes=10))]
        d.update(asw_took=bool(len(am)), lp_took=bool(len(pm)), live_open_n=nopen, live_maxopen=mo_at(y.t))
        ag = pg = ""
        if a_t:
            Lr = nearest_lines(J["rows"][a_t[0]], y.pair, td, -200, 60)
            f = Lr[(Lr.e == "FAILS") & Lr.dir.isin(["LONG"])]
            ag = f.iloc[(f.dts.abs()).argmin()].gate if len(f) else ""
            if not ag:
                sh = Lr[(Lr.e == "FAILS") & (Lr.dir == "SHORT")]
                ag = "DIR_SHORT" if len(sh) else ("NO_LINE" if not len(Lr) else "OTHER:" + ",".join(sorted(set(Lr.gate)))[:60])
        if p_t:
            Lr = nearest_lines(J["rows"][p_t[0]], y.pair, td, -200, 60)
            f = Lr[(Lr.e == "FAILS") & Lr.dir.isin(["LONG"])]
            pg = f.iloc[(f.dts.abs()).argmin()].gate if len(f) else ""
        d.update(asw_gate=ag, lp_gate=pg)
        ljg = ""
        if pd.notna(lj0) and y.t >= lj0:
            q = LJ[(LJ.pair == y.pair) & ((LJ.t - td).dt.total_seconds().between(-200, 60)) & (LJ.e.isin(["BLOCK", "FAILS"]))]
            q = q[q.dir.astype(str).isin(["LONG", "nan", ""])] if "dir" in q else q
            ljg = ",".join(sorted(set(q.gate.dropna().astype(str))))[:120] if len(q) else "NO_LINE"
        d["live_journal"] = ljg
        if len(lp_):
            c, det = "LIVE_PROBE", str(lp_.iloc[0].cell_multiplier_source)
        elif len(held):
            c, det = "LIVE_PAIR_HELD", str(held.iloc[0].entry_strategy)
        elif nopen >= d["live_maxopen"]:
            c, det = "LIVE_SLOTS_FULL", f"{nopen}/{d['live_maxopen']}"
        elif len(oth):
            c, det = "LIVE_OTHER_TIME", f"{(oth.t - y.t).dt.total_seconds().abs().min() / 60:.0f} min"
        elif not a_t:
            c, det = "NO_ASW", ""
        elif len(am):
            c, det = "ASW_TOOK", f"asw {am.iloc[0].pct:+.2f}"
        elif len(pm):
            c, det = "RULE_THEN", ag
        else:
            c, det = "PHASE", ag
        d.update(cls=c, detail=det)
        ex.append(d)
    EX = pd.DataFrame(ex)
    EX.to_csv(os.path.join(REP, "study_ml_trace_extras.csv"), index=False)
    # console
    lo = SR[SR.cls != "MATCHED"]
    print(f"seed rows {len(SR)} · matched {int((SR.cls == 'MATCHED').sum())} · live-only seed-rows {len(lo)}")
    print(lo.groupby(lo.cls).agg(n=("pair", "size"), live=("pct_live", "mean")).sort_values("n", ascending=False).to_string())
    print("tick source (all seed rows):", SR.src.value_counts().to_dict())
    print(EX.groupby("cls").agg(n=("pair", "size"), rep=("pct_rep", "mean")).sort_values("n", ascending=False).to_string())


if __name__ == "__main__":
    main()
