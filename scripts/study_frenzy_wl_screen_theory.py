#!/usr/bin/env python3
"""FRENZY W/L screen 2026-10-19 — theory-driven family (mechanism hypotheses, predicted direction fixed in the code BEFORE reading
their results) + existing observe-line tallies on all real fills. Read-only; writes reports/FRENZY_WL_SCREEN_2026-10-19_theory.csv.
argv[1] = real-fill pool pickle. Cut = sign for signed macro vars, else the REAL-pool median, frozen and re-applied to the backtest."""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_yr6wl_frenzy_common as C
import study_frenzy_wl_screen as W

H = {  # sleeve: [(id, var, zone side, predicted zone effect, mechanism)]
    "FRENZY_LONG": [("L1", "entry_rsi", "hi", "worse", "exhaustion: buying an overbought ON bar, +1.5 must come before mean reversion"),
                    ("L2", "entry_adx", "hi", "worse", "late trend: high ADX = the move is mature"),
                    ("L3", "entry_frenzy_vs_vwap_pct", "hi", "worse", "stretch above the spike VWAP = less room, more give-back"),
                    ("L4", "entry_btc_rsi", "hi", "worse", "BTC 5m hot → market-wide pullback drags alts"),
                    ("L5", "x_log_vol24", "lo", "worse", "thin pair (non-top-50 proxy) → stop runs / gaps"),
                    ("L6", "entry_btc_1d_ret_pct", "neg", "worse", "BTC down day → alt pumps fade")],
    "FRENZY_LITE": [("T1", "entry_frenzy_above_share", "lo", "worse", "weak VWAP hold since the spike = distribution"),
                    ("T2", "entry_btc_1h_slope", "neg", "worse", "BTC falling intraday (existing LITE_BTC1H_FALLING)"),
                    ("T3", "entry_pair_ema20_ema50_gap_pct", "hi", "worse", "stretched re-entry (existing LITE_STRETCH_REENTRY family)"),
                    ("T4", "entry_frenzy_vol_trend", "lo", "worse", "volume fading → the second leg lacks fuel"),
                    ("T5", "entry_frenzy_hours", "hi", "worse", "stale episode → attention gone")],
    "FRENZY_WILLY": [("W1", "entry_vol_mcap_ratio", "hi", "worse", "turnover high inside the R < 1 band → churn, fragile bid"),
                     ("W2", "x_log_mcap", "lo", "worse", "small cap → crash fills (yr6 2025 non-top-50 losses)"),
                     ("W3", "entry_btc_1h_slope", "neg", "worse", "BTC falling (existing WILLY_BTC1H)"),
                     ("W4", "entry_frenzy_bar_ret_pct", "lo", "better", "deep first red bar = better dip price (DL 279 DIP_SUB)"),
                     ("W5", "entry_atr_pct", "hi", "worse", "high pair ATR → −4 reached by noise")],
}


def main():
    D = pd.read_pickle(sys.argv[1]); B = W.load_bt()
    rows = []
    for sl, hs in H.items():
        S = D[D.sleeve == sl]
        for hid, var, side, pred, mech in hs:
            x = W.numcol(S[var]); y = S.y.values
            cut = 0.0 if side == "neg" else float(np.nanmedian(x))
            rule = ("<=", cut) if side in ("lo", "neg") else (">", cut)
            if side == "neg":
                rule = ("<=", 0.0)
            z, sc = W.apply_rule(x, rule)
            st = W.stats_zone(y, z, sc, S.day.values, S.pair.values)
            sgn = -1 if pred == "worse" else 1
            real_ok = (st["avg"] - st["rest_avg"]) * sgn > 0
            r = dict(sleeve=sl, id=hid, var=var, rule=f"{rule[0]} {rule[1]:.4g}", predicted=pred, mechanism=mech,
                     real=f"{st['N']}·{st['days']}d·{st['WR']}%·{st['avg']:+.3f} vs {st['rest_N']}·{st['rest_WR']}%·{st['rest_avg']:+.3f}",
                     real_dir_ok=real_ok, real_P_pred=round(C.boot_diff_p(y[sc], z[sc], S.day.values[sc], sgn), 3))
            bt = W.bt_check(B, sl, var, rule)
            ok = 0
            for b in bt:
                if "avg" in b and "top50" not in b["regime"]:
                    good = (b["avg"] - b["rest_avg"]) * sgn > 0
                    r[f"BT {b['regime']}"] = f"{b['N']}·{b['WR']}%·{b['avg']:+.3f} vs {b['rest_avg']:+.3f} (P {1 - b['P_zone_lt_rest'] if sgn > 0 else b['P_zone_lt_rest']:.2f})"
                    if b["regime"] != "ALL":
                        ok += good
                elif "avg" in b:
                    r[f"BT {b['regime']}"] = f"{b['N']}·{b['avg']:+.3f} vs {b['rest_avg']:+.3f}"
            r["BT_regimes_pred_dir"] = f"{ok}/3"
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(ROOT, "reports", "FRENZY_WL_SCREEN_2026-10-19_theory.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    print(T.drop(columns=["mechanism"]).to_string())
    # existing observe lines — descriptive tally on ALL real fills (each line's own gate counts from its deploy, via its scout)
    out = []
    def tally(name, sl, mask, sc):
        S = D[D.sleeve == sl]; y = S.y.values
        st = W.stats_zone(y, mask, sc, S.day.values, S.pair.values)
        out.append(dict(line=name, sleeve=sl, **{k: st[k] for k in ("N", "days", "WR", "avg", "sum", "rest_N", "rest_WR", "rest_avg", "unscored")}))
    S = D[D.sleeve == "FRENZY_LITE"]
    x = W.numcol(S.entry_btc_1h_slope); tally("LITE_BTC1H_FALLING (slope ≤ 0)", "FRENZY_LITE", (x <= 0).values & x.notna().values, x.notna().values)
    g = W.numcol(S.entry_pair_ema20_ema50_gap_pct); k = S.x_kth_in_episode.values
    tally("LITE_STRETCH_REENTRY (gap > 1.87 ∧ k ≥ 2, k = all FRENZY fills incl. WILLY — approx.)", "FRENZY_LITE", ((g > 1.87).values & (k >= 2)), g.notna().values)
    gv = W.numcol(S.entry_frenzy_gvol); tally("GVOL_SPLIT LITE (entry_frenzy_gvol ≥ 1.0)", "FRENZY_LITE", (gv >= 1.0).values, gv.notna().values)
    S = D[D.sleeve == "FRENZY_WILLY"]
    x = W.numcol(S.entry_btc_1h_slope); tally("WILLY_BTC1H blocked side (slope ≤ 0)", "FRENZY_WILLY", (x <= 0).values & x.notna().values, x.notna().values)
    b = W.numcol(S.entry_frenzy_bar_ret_pct); tally("WILLY DIP_SUB deep red bar (≤ −1.16), all slopes", "FRENZY_WILLY", (b <= -1.16).values, b.notna().values)
    S = D[D.sleeve == "FRENZY_LONG"]
    gv = W.numcol(S.entry_frenzy_gvol); tally("GVOL_SPLIT LONG (entry_frenzy_gvol ≥ 1.0)", "FRENZY_LONG", (gv >= 1.0).values, gv.notna().values)
    O = pd.DataFrame(out); O.to_csv(os.path.join(ROOT, "reports", "FRENZY_WL_SCREEN_2026-10-19_existing_lines.csv"), index=False)
    print(O.to_string())


if __name__ == "__main__":
    main()
