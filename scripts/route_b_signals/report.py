"""Route B signal-level loader — the markdown report (coverage · data quality · parity · existing-sleeve descriptive tables · H1–H6
COUNTS ONLY). The H1–H6 outcome columns are never read here (pre-registration first): only identity / state-coverage / h_final columns."""
import os

import numpy as np
import pandas as pd

import common as C

KNOWN_BLOCK_PREFIX = ("EXH_", "DIST_", "FRENZY_WILLY", "FLATBASE_")


def _pct(a, b):
    return f"{100 * a / b:.0f} %" if b else "–"


def _t(df):
    if not len(df):
        return "_(none)_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(str(c) for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    sign = [("pct" in str(c) or c == "delta") for c in cols]
    for r in df.itertuples(index=False):
        out += "| " + " | ".join("" if (isinstance(v, float) and v != v) else
                                 ((f"{v:+.2f}" if sg else f"{v:g}") if isinstance(v, float) else str(v)) for v, sg in zip(r, sign)) + " |\n"
    return out


def coverage(S):
    g = S.groupby(["sleeve", "status"], dropna=False)
    rows = []
    for (sl, st), x in g:
        sole = int(x.sole_block.fillna(False).astype(bool).sum()) if "sole_block" in x.columns else 0
        today_ref = int((x.today_block.fillna("").astype(str) != "").sum()) if "today_block" in x.columns else 0   # (635)
        pr = x.get("out_priced", pd.Series(False, index=x.index)).fillna(False).astype(bool)
        fin = x.get("out_final", pd.Series(False, index=x.index)).fillna(False).astype(bool)
        rows.append(dict(sleeve=sl, status=st, block_class=(x.block_class.iloc[0] if "block_class" in x.columns else ""), signals=len(x),
                         possibilities=int((~x.poss_dup.astype(bool)).sum()), sole_block=sole, today_stack_refuses=today_ref, windows15=x.window15.nunique(),
                         coins=x.pair.nunique(), book_at_signal=_pct(int(x.rb_has_book.fillna(False).astype(bool).sum()), len(x)),
                         mark_at_signal=_pct(int(x.rb_has_mark.fillna(False).astype(bool).sum()), len(x)),
                         priced=_pct(int(pr.sum()), len(x)), final=_pct(int((pr & fin).sum()), len(x))))
    return pd.DataFrame(rows)


def descriptive(S):
    """existing sleeves: TAKEN (linked fills) vs SOLE-BLOCK filter rows (the named gate is the only failing filter and every filter was
    scored) — possibilities only (poss_dup False), priced FINAL rows. Capacity (PAIR_HELD / SLEEVE_OFF), core-rule misses, multi-gate
    blocks, partly-unscored blocks and unlinked taken signals are left out (counted in section 2)."""
    x = S[(~S.poss_dup.astype(bool)) & S.get("out_priced", False).fillna(False).astype(bool) & S.get("out_final", False).fillna(False).astype(bool)].copy()
    x = x[(x.status == "OPENED") | ((x.block_class == "filter") & x.sole_block.astype(bool))]
    tb = x["today_block"].fillna("").astype(str) if "today_block" in x.columns else pd.Series("", index=x.index)
    x["group"] = np.where(x.status != "OPENED", x.status + " (sole)",
                          np.where(tb != "", "TAKEN then · today's stack refuses: " + tb, "TAKEN"))   # (635) filters armed mid-recording
    rows = []
    for (sl, gr), y in x.groupby(["sleeve", "group"]):
        n = y.out_net_pct
        rows.append(dict(sleeve=sl, group=gr, N=len(y), windows15=y.window15.nunique(), coins=y.pair.nunique(), WR=_pct(int((n > 0).sum()), len(y)),
                         avg_net_pct=float(n.mean()), median_net_pct=float(n.median()), TP=int((y.out_reason == "TP").sum()),
                         STOP=int((y.out_reason == "STOP_LOSS").sum()), TIME=int((y.out_reason == "MAX_HOLD_TIME").sum()),
                         avg_mfe_pct=float(y.out_mfe_pct.mean()), avg_mae_pct=float(y.out_mae_pct.mean())))
    return pd.DataFrame(rows)


def parity(S):
    x = S[(S.status == "OPENED") & S.fill_pnl_pct.notna() & (S.fill_status == "CLOSED") & S.get("out_final", False).fillna(False).astype(bool)
          & ~S.fill_operator_closed.fillna(False).astype(bool)]
    if not len(x):
        return pd.DataFrame()
    d = pd.DataFrame(dict(sleeve=x.sleeve, pair=x.pair, opened=x.fill_opened_at.str[:16], booked_reason=x.fill_close_reason,
                          walker_reason=x.out_reason, booked_net_pct=x.fill_pnl_pct, walker_net_pct=x.out_net_pct,
                          delta=x.out_net_pct - x.fill_pnl_pct, walker_funding_pct=x.out_funding_pct))
    return d.sort_values("opened")


def write(S, Tt, J, meta, book, mark):
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    path = os.path.join(C.REPORTS, f"ROUTE_B_SIGNALS_{day}.md")
    L = []
    hours = (pd.Timestamp(meta["data_end"]) - C.REC_START).total_seconds() / 3600
    days = hours / 24
    L.append(f"# 📡 Route B — signal-level dataset ({day})\n")
    pr = S.get("out_priced", pd.Series(False, index=S.index)).fillna(False).astype(bool)
    fi = S.get("out_final", pd.Series(False, index=S.index)).fillna(False).astype(bool)
    L.append(f"**Short version.** Every signal the bot journaled since Route B recording began (2026-11-19 23:23 UTC) is one row: "
             f"**{len(S)} signals** ({int((~S.poss_dup).sum())} possibilities, {S.window15.nunique()} market 15-min windows) across "
             f"{S.sleeve.nunique()} sleeves, {int(S.rb_has_book.fillna(False).astype(bool).sum())} with the order book recorded at the signal "
             f"({_pct(int(S.rb_has_book.fillna(False).astype(bool).sum()), len(S))}), {int(pr.sum())} priced under today's exit "
             f"({int((pr & fi).sum())} final). Window: {hours:.1f} h, one BTC regime. **Everything here is descriptive; N is tiny; nothing "
             f"is a ship input.** H1–H6 trigger outcomes are stored but deliberately NOT shown (pre-registration first).\n")
    L.append("Research only: no engine / config / recorder change, S3 read-only. Code: `scripts/route_b_signals/` (usage in `build.py`'s "
             "docstring). Datasets: `reports/ROUTE_B_SIGNALS_data/` (gitignored).\n")
    L.append("## 1 · Inputs and the exits priced\n")
    L.append(_t(pd.DataFrame([
        dict(item="Route B per-minute summaries", value=f"{meta['s3_hours']} hourly files · data to {meta['data_end'][:16]} UTC"),
        dict(item="Decision journal", value=f"{len(meta['journal_files'])} exports read · last journal row {meta['journal_end'][:19]} UTC "
                                            f"(signals after it are invisible until the next export; taken fills are added from the orders export: "
                                            f"{meta['orders_only_fills']})"),
        dict(item="Probe fills dropped", value=str(meta["probe_fills_dropped"])),
        dict(item="Costs", value=f"taker {C.TAKER_PCT:g} % (live config) + modelled slip per taker leg (recorded book, $6k walk) · maker TP {C.MAKER_PCT:g} % "
                                 f"(fills only {C.TP_THROUGH_PCT} % through the level) · real settled funding (fapi /fundingRate)"),
        dict(item="Entry", value="the open of the first minute starting ≥ signal + 1 s. EXH / DIST signal = the 15m bar CLOSE, so blocked / "
                                 "refused rows enter at bar close + 60 s, while the bot's real fills land ~30–40 s after the close (blocked rows "
                                 "enter ~20–30 s later than a real fill would) · H1–H6: the open of minute m + 2 after trigger minute m · taken "
                                 "signals: the REAL fill price (flagged real_fill; walk starts the minute after the fill; paper drift inside the fill)"),
        dict(item="Taken fill minute", value="walked PESSIMISTICALLY: open = the fill price, the minute's whole adverse extreme kept (it may come "
                                             "after the fill), its favourable extreme capped at the better of fill and close (out_fill_minute)"),
        dict(item="Stop fill", value="a minute opening beyond the stop fills at the open; otherwise at the WORSE of the level and the minute's close "
                                     "(pessimistic on purpose — the bot polls every few seconds, real stops land between the level and this price)"),
    ])))
    ex = []
    for sl, sp in meta["exits"].items():
        if sp is None:
            ex.append(dict(sleeve=sl, exit="NOT PRICED — " + C.NOT_PRICED_WHY.get(sl, "")))
        elif sp["kind"] == "price":
            ex.append(dict(sleeve=sl, exit=f"TP −{sp['tp']:g} % maker · stop +{sp['sl']:g} %{' incl. funding paid' if sp['stop_counts_funding'] else ''} · {sp['hold_min'] // 60} h"))
        elif sp["kind"] == "atr":
            ex.append(dict(sleeve=sl, exit=f"TP min({sp['tp_atr']:g}×ATR, {sp['tp_cap']:g}) % maker · SL min({sp['sl_atr']:g}×ATR, {sp['sl_cap']:g}) % · "
                                           f"{sp['hold_min'] // 60} h (ATR: journal stamp, else Wilder ATR14 on 15m closed ≤ the signal)"))
        else:
            ex.append(dict(sleeve=sl, exit=f"TP +{sp['tp_net']:g} % net maker · stop −{sp['sl_net']:g} % net · {sp['hold_min'] // 60} h"))
    L.append("\nExits read from the live `trading_config.json` at run time (EXHAUSTION TP cap is 7 since DL 609 — the journaled "
             "exh_tp_pct = 10 on these rows is the pre-609 level and is NOT used; the journaled ATR is):\n")
    L.append(_t(pd.DataFrame(ex)))
    L.append("\n## 2 · Coverage — signals per sleeve × status\n")
    L.append("possibilities = one per sleeve × coin per holding period (a later signal inside a possibility is flagged `poss_dup`, not "
             "dropped) · book / mark at signal = the recorded state existed at the state minute (no look-ahead) · priced = an outcome "
             "under today's exit · final = the path finished.\n")
    L.append(_t(coverage(S)))
    # ── data quality
    L.append("\n## 3 · Data-quality flags\n")
    q = []
    p = S[pr]
    for c in ("out_entry_slip_src", "out_exit_slip_src", "out_cost_flag", "out_entry_src", "out_atr_src"):
        if c in p.columns:
            q.append(dict(flag=c, value=" · ".join(f"{k}: {v}" for k, v in p[c].fillna("–").replace("", "–").value_counts().items())))
    hb = S[S.rb_has_book.fillna(False).astype(bool)]
    if len(hb):
        for tag in ("05", "1"):
            lb = (hb[f"rb_bid_usd_{tag}_lb"].astype(bool) | hb[f"rb_ask_usd_{tag}_lb"].astype(bool)).mean()
            q.append(dict(flag=f"depth within {0.5 if tag == '05' else 1} % is a LOWER bound (20 levels reach less)", value=f"{lb * 100:.0f} % of signals with a book"))
        q.append(dict(flag="$6k cost not computable (20 levels too shallow)",
                      value=f"{(hb.rb_cost6k_buy_lb.astype(bool) | hb.rb_cost6k_sell_lb.astype(bool)).mean() * 100:.0f} % of signals with a book"))
    if "out_atr_check_computed" in S.columns:
        z = S[S.out_atr_check_computed.notna() & S.out_atr_pct.notna()]
        if len(z):
            rel = (z.out_atr_check_computed / z.out_atr_pct - 1).abs()
            q.append(dict(flag="ATR check: computed 15m Wilder ATR vs the journal stamp",
                          value=f"N {len(z)} · median |rel diff| {rel.median() * 100:.1f} % · max {rel.max() * 100:.1f} %"))
    if "out_funding_ok" in p.columns:
        q.append(dict(flag="funding history fetched", value=f"{int(p.out_funding_ok.fillna(False).astype(bool).sum())} of {len(p)} priced"))
    q.append(dict(flag="not final (path not finished · funding not fetched · unlinked taken fill) — re-priced next run", value=str(int((pr & ~fi).sum()))))
    q.append(dict(flag="taken signals whose fill is in no orders export yet (OPENED_UNLINKED: out of TAKEN / parity, never final)",
                  value=str(int((S.status == "OPENED_UNLINKED").sum()))))
    q.append(dict(flag="fills closed by the operator / backstop (fill_operator_closed: out of parity)",
                  value=str(int(S.fill_operator_closed.fillna(False).astype(bool).sum()))))
    if "block_class" in S.columns:
        bl = S[S.block_class == "filter"]
        q.append(dict(flag="filter blocks: sole (only failing gate, all scored) · also blocked by another gate · gates unscored (fail-fast engine)",
                      value=f"{int(bl.sole_block.astype(bool).sum())} · {int((bl.also_blocked_by.astype(str) != '').sum())} · "
                            f"{int((bl.filters_unevaluated.astype(str) != '').sum())} of {len(bl)}"))
        q.append(dict(flag="EXH_SSC_OIVOL not reached on earlier EXH blocks (the engine reads OI only after every other gate passed) → "
                           "an earlier EXH block can never be a clean single-filter row: the EXH sole set is STRUCTURALLY limited to SSC_OIVOL blocks",
                      value=str(int((S.get("filters_not_reached", pd.Series("", index=S.index)).astype(str) != "").sum()))))
        q.append(dict(flag="block classes", value=" · ".join(f"{k}: {v}" for k, v in S.block_class.value_counts().items())))
        q.append(dict(flag="PAIR_HELD rows (capacity: joined to the holding position's possibility, not a filter counterfactual)",
                      value=str(int(S.status.astype(str).str.endswith("PAIR_HELD").sum()))))
    if "today_recheck" in S.columns:   # (635) filters armed after the recording began, re-applied to the signals before them
        for nm, y in S[S.today_recheck.fillna("").astype(str) != ""].groupby("today_recheck"):
            vc = y.today_recheck_state.value_counts()
            q.append(dict(flag=f"today's stack: {nm} re-applied to signals from before it went live (zone = today refuses · rest · unread = "
                               f"fail-open · nofetch = not read this build, unknown)",
                          value=" · ".join(f"{k} {int(vc.get(k, 0))}" for k in ("zone", "rest", "unread", "nofetch")) + f" of {len(y)}"))
    q.append(dict(flag="EXH_WIDE_READ_CAP counters (shortlist coins not read: no episode, not signals)", value=str(meta["exh_wide_read_cap"])))
    q.append(dict(flag="FLATBASE episodes still TRACKING (no final status yet)", value=str(meta["fb_tracking_only"])))
    if len(J):
        B = J[J.e == "BLOCK"]
        unk = B[~B.gate.astype(str).str.startswith(KNOWN_BLOCK_PREFIX) & ~B.gate.astype(str).str.match(r"^FRENZY_[A-Z_]+_OFF$")]
        q.append(dict(flag="BLOCK gates not mapped to a sleeve", value=(" · ".join(f"{k}: {v}" for k, v in unk.gate.value_counts().head(8).items()) or "none")))
    q.append(dict(flag="core rule not met (DIST_NOT_VSW pre-screen, FB NOT_FLAT, FB no-confirm, WILLY no entry bar)",
                  value=f"{int((~S.core_rule_met.astype(bool)).sum())} rows — keep them apart from real signals in any read"))
    q.append(dict(flag="Binance API stopped (418/429)", value=str(meta["binance_stopped"])))
    L.append(_t(pd.DataFrame(q)))
    # ── parity
    L.append("\n## 4 · Parity — the walker vs the booked P&L of the taken fills\n")
    L.append("Same entry (the real fill), today's exit, recorded costs and real funding. A large Δ means the walker or the exit spec is "
             "wrong for that sleeve (the booked P&L reflects the exit the fill RAN, e.g. EXH at TP 10 before DL 609, HT at TP 5 before the "
             "+6 change) — read Δ with the exit era in mind.\n")
    L.append(_t(parity(S)))
    # ── descriptive
    L.append("\n## 5 · Existing sleeves — taken vs blocked / refused (DESCRIPTIVE · N-tiny · one regime · NOT a ship input)\n")
    L.append("Possibilities only, priced final rows. TAKEN = linked fills; blocked rows ONLY where the named gate was the SOLE failing filter "
             "and every filter was scored (the engine stops at the first failing gate, so most blocked rows are multi-gate or partly unscored "
             "— see section 3); capacity rows (PAIR_HELD) are not filter counterfactuals. Blocked rows are priced as if entered at the "
             "next-minute open with today's exit — they are what each gate kept the bot out of in THIS window, nothing more. A fill taken before a "
             "filter armed mid-recording (DL 601 / 619) that today's stack would refuse is its own row ('TAKEN then · today's stack refuses'). "
             "Window units decide any later read (the locked "
             "gates: ≥ 15 blocked, ≥ 8 windows, window-clustered bootstrap); no row here meets them.\n")
    L.append(_t(descriptive(S)))
    # ── H1–H6 counts only
    L.append("\n## 6 · New-sleeve triggers H1–H6 — COUNTS AND COVERAGE ONLY (outcomes stored, not read)\n")
    L.append("Definitions: Route B report §4, made causal (each coin's 99th pct / median from rows BEFORE the trigger minute, so counts differ "
             "from §4's whole-sample counts). One possibility per coin per 60 min (H6: 24 h), alts only. The forward returns (5 / 15 / 60 / "
             "240 min) and the TP/SL grid are in `triggers_H.csv.gz`; open them only after the PREREG file is frozen.\n")
    if len(Tt):
        hrow = []
        for (k, d), y in Tt.groupby(["trigger", "direction"]):
            hrow.append(dict(trigger=k, direction=d, possibilities=len(y), coins=y.pair.nunique(), windows15=y.window15.nunique(),
                             per_day=round(len(y) / days, 1), windows_per_day=round(y.window15.nunique() / days, 1),
                             book_at_trigger=_pct(int(y.rb_has_book.fillna(False).astype(bool).sum()), len(y)),
                             full_240m_path=_pct(int(y.h_final.fillna(False).astype(bool).sum()), len(y))))
        L.append(_t(pd.DataFrame(hrow)))
    L.append("\n## 7 · Daily run · known gaps\n")
    L.append("- Run: `venv/bin/python -I scripts/route_b_signals/build.py` after the day's dashboard exports are downloaded (incremental: "
             "final outcomes carried over, censored ones re-priced, new S3 hours / klines days fetched). `--full` re-prices everything.\n"
             "- **Journal lag:** signals exist only up to the newest decisions export; the bot does not push its journal to S3.\n"
             "- **Plain FRENZY_WILLY not priced** (V2b spike-high target / late TP / runner need inputs not journaled); FRENZY_WIDE is off.\n"
             "- **Unpriced by design:** FB NOT_FLAT / VOL24_LOW and WILLY RED_EXPIRED / NOMISS_MISSED have no entry bar.\n"
             "- **Taken fills** walk from the minute after the fill (the rest of the fill minute is not walked); paper fills carry paper drift.\n"
             "- **1-minute bars:** a minute touching both levels books the stop; intraminute order unknown (conservative).\n"
             "- **20-level book:** depth / cost on fine-tick coins are lower bounds or missing (the default 0.10 % slip is then used and flagged).\n"
             "- **Liquidations are a sample** (≤ 1 per coin per second); the burst baseline needs ≥ 20 prior liquidation minutes per coin.\n"
             "- **One regime, ~2 days**: every table is descriptive until the pre-registered weekly scan (≈ 2026-12-10).\n")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path
